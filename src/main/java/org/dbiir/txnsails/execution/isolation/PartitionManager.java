package org.dbiir.txnsails.execution.isolation;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.dbiir.txnsails.common.constants.SmallBankConstants;
import org.dbiir.txnsails.common.constants.TPCCConstants;
import org.dbiir.txnsails.common.constants.YCSBConstants;
import org.dbiir.txnsails.common.types.IsolationLevelType;
import org.dbiir.txnsails.execution.validation.ValidationMeta;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class PartitionManager {
  private static Logger logger = LoggerFactory.getLogger(PartitionManager.class);
  private static final PartitionManager INSTANCE;
  private boolean staticConfig = false;
  private final HashMap<String, ArrayList<LinkedList<DataItem>>> tableToDataItems;
  private final HashMap<String, ArrayList<ReentrantReadWriteLock>> tableToDataItemGuards;

  // allocate partition config via micro partitions
  private PartitionConfig partitionConfig;
  private final HashMap<String, Integer> relationToPartitionSize;
  private final HashMap<String, List<PartitionInfo>> relationToPartitions;
  @Getter private String workload;
  private long startTime = 0L;

  // cross transaction statistic
  private final HashMap<HashMap<Integer, Integer>, Integer> partitionRelations = new HashMap<>();

  static {
    INSTANCE = new PartitionManager();
  }

  public PartitionManager() {
    this.tableToDataItems = new HashMap<>();
    this.tableToDataItemGuards = new HashMap<>();
    this.relationToPartitionSize = new HashMap<>();
    this.relationToPartitions = new HashMap<>();
  }

  public void init(String workload, String configFile) {
    this.staticConfig = true;
    try {
      this.partitionConfig = new PartitionConfig().load(configFile);
      for (PartitionConfig.Relation relationConfig : this.partitionConfig.getRelations()) {
        this.relationToPartitionSize.put(
            relationConfig.getName(), relationConfig.getPartitionSize());
        this.relationToPartitions.put(relationConfig.getName(), new ArrayList<>());
      }
      for (PartitionConfig.Partition partition : this.partitionConfig.getPartitions()) {
        this.relationToPartitions
            .get(partition.getRelationName())
            .add(new PartitionInfo(partition.getId()));
      }
    } catch (IOException e) {
      this.staticConfig = false;
      logger.error("Error loading configuration file", e);
    }

    init(workload);
  }

  public void init(String workload) {
    this.workload = workload;
    switch (workload) {
      case "ycsb":
        for (Map.Entry<String, Integer> entry : YCSBConstants.TABLENAME_TO_HASH_SIZE.entrySet()) {
          if (entry.getValue() <= 0) {
            continue;
          }
          createItemTable(entry.getKey(), entry.getValue());
        }
        break;
      case "smallbank":
        for (Map.Entry<String, Integer> entry :
            SmallBankConstants.TABLENAME_TO_HASH_SIZE.entrySet()) {
          if (entry.getValue() <= 0) {
            continue;
          }
          createItemTable(entry.getKey(), entry.getValue());
        }
        break;
      case "tpcc":
        logger.error("not implemented");
        break;

      default:
        throw new RuntimeException("Unsupported workload: " + workload);
    }
  }

  private void createItemTable(String relationName, int bucketSize) {
    tableToDataItemGuards.put(relationName, new ArrayList<>(bucketSize));
    tableToDataItems.put(relationName, new ArrayList<>(bucketSize));
    for (int i = 0; i < bucketSize; i++) {
      tableToDataItemGuards.get(relationName).add(new ReentrantReadWriteLock());
      LinkedList<DataItem> dataItems = new LinkedList<>();
      dataItems.add(new DataItem(i, getPartitionId(relationName, i), relationName));
      tableToDataItems.get(relationName).add(dataItems);
    }
  }

  /*
   * @return isolation level: `0-serializable`, 1-`snapshot isolation`; `2-read committed`
   */
  public int chooseIsolation(ValidationMeta validationMeta) {
    if (startTime == 0L) {
      startTime = System.currentTimeMillis();
    }

    int partitionId = getPartitionId(validationMeta);
    IsolationLevelType isolationLevelType = IsolationLevelType.SER;
    if (staticConfig) {
      isolationLevelType =
          partitionConfig.getIsolationLevel(System.currentTimeMillis() - startTime, partitionId);
    } else {
      isolationLevelType =
          relationToPartitions.get(validationMeta.getRelationName()).get(partitionId).getLevel();
    }

    switch (isolationLevelType) {
      case SER -> {
        return 0;
      }
      case SI -> {
        return 1;
      }
      case RC -> {
        return 2;
      }
      default -> {
        logger.error("Unknown isolation level");
        return -1;
      }
    }
  }

  public int getMu(int partitionId, String relationName) {
    if (staticConfig) {
      return 2;
    } else {
      return relationToPartitions.get(relationName).get(partitionId).getMu();
    }
  }

  /*
   * this.relationToPartition:
   *    - relation name -> list of partition ids, e.g., R1 -> [0,1,2], R2 -> [3,4,5]
   * this.relationToPartitionSize:
   *    - relation name -> partition size, e.g., R1 -> 1000, R2 -> 5000
   *
   * {table: R2, key: 5001}
   * partitionId is calculated by
   *    - offsetPartition: key / partition size = 5001 / 5000 = 1
   *    - partitionId: this.relationToPartitions.get(R2).get(1) = 4
   */
  public int getPartitionId(ValidationMeta validationMeta) {
    int key = validationMeta.getIdForValidation();
    int partitionSize = this.relationToPartitionSize.get(validationMeta.getRelationName());
    int offsetPartition = key / partitionSize;
    return this.relationToPartitions
        .get(validationMeta.getRelationName())
        .get(offsetPartition)
        .getPartitionId();
  }

  public int getPartitionId(String relationName, int key) {
    int partitionSize = this.relationToPartitionSize.get(relationName);
    int offsetPartition = key / partitionSize;
    return this.relationToPartitions.get(relationName).get(offsetPartition).getPartitionId();
  }

  public DataItem getAndAddDataItem(ValidationMeta validationMeta) {
    int bucketNum =
        validationMeta.getIdForValidation()
            % getHashSizeByRelationName(validationMeta.getRelationName());
    ReadWriteLock lock = tableToDataItemGuards.get(validationMeta.getRelationName()).get(bucketNum);
    lock.readLock().lock();
    for (DataItem item : tableToDataItems.get((validationMeta.getRelationName())).get(bucketNum)) {
      if (item.getKey() == validationMeta.getIdForValidation()) {
        lock.readLock().unlock();
        return item;
      }
    }
    lock.readLock().unlock();
    int partitionId = getPartitionId(validationMeta);
    lock.writeLock().lock();
    DataItem item =
        new DataItem(
            validationMeta.getIdForValidation(), partitionId, validationMeta.getRelationName());
    tableToDataItems.get((validationMeta.getRelationName())).get(bucketNum).add(item);
    lock.writeLock().unlock();
    return item;
  }

  // TODO:
  public void collect(Transaction transaction) {
    for (Participant participant : transaction.getParticipants()) {}
  }

  private int getHashSizeByRelationName(String relationName) {
    return switch (this.workload) {
      case "ycsb" -> YCSBConstants.getHashSize(relationName);
      case "smallbank" -> SmallBankConstants.getHashSize(relationName);
      case "tpcc" -> TPCCConstants.getHashSize(relationName);
      default -> {
        System.out.println("Unknown relation");
        yield -1;
      }
    };
  }

  public static PartitionManager getInstance() {
    return INSTANCE;
  }

  @Getter
  @Setter
  @NoArgsConstructor
  @AllArgsConstructor
  private class PartitionInfo {
    private int partitionId;
    private Integer mu;
    private IsolationLevelType level;
    private float workloadIntensive;

    public PartitionInfo(int partitionId) {
      this.partitionId = partitionId;
      this.mu = 2;
      this.level = IsolationLevelType.SER;
      this.workloadIntensive = 0.0F;
    }
  }
}
