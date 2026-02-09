package org.dbiir.txnagent.execution.isolation;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.dbiir.txnagent.common.constants.SmallBankConstants;
import org.dbiir.txnagent.common.constants.TPCCConstants;
import org.dbiir.txnagent.common.constants.YCSBConstants;
import org.dbiir.txnagent.common.types.IsolationLevelType;
import org.dbiir.txnagent.execution.validation.ValidationMeta;
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
  private final HashMap<String, Integer> relationToPartitionCount;
  private final ArrayList<PartitionInfo> partitions; // partitionId -> PartitionInfo
  @Getter private String workload;
  private long startTime = 0L;

  static {
    INSTANCE = new PartitionManager();
  }

  public PartitionManager() {
    this.tableToDataItems = new HashMap<>();
    this.tableToDataItemGuards = new HashMap<>();
    this.relationToPartitionSize = new HashMap<>();
    this.relationToPartitionCount = new HashMap<>();
    this.partitions = new ArrayList<>(8);
  }

  public void init(String workload, String configFile) {
    this.staticConfig = true;
    try {
      this.partitionConfig = new PartitionConfig().load(configFile);
      for (PartitionConfig.Relation relationConfig : this.partitionConfig.getRelations()) {
        this.relationToPartitionSize.put(
            relationConfig.getName(),
            relationConfig.getPartitionSize()); // how many data items in one partition
        this.relationToPartitionCount.put(
            relationConfig.getName(),
            relationConfig.getPartitionCount()); // how many partitions in one relation
      }
      int partitionCount = -1;
      switch (workload) {
        case "ycsb" -> {
          YCSBConstants.loadPartitioningInfo(
              this.relationToPartitionCount, this.relationToPartitionSize);
          partitionCount = YCSBConstants.getGlobalPartitionIds().size();
        }
        case "smallbank" -> {
          SmallBankConstants.loadPartitioningInfo(
              this.relationToPartitionCount, this.relationToPartitionSize);
          partitionCount = SmallBankConstants.getGlobalPartitionIds().size();
        }
        case "tpcc" -> {
          TPCCConstants.loadPartitioningInfo(
              this.relationToPartitionCount, this.relationToPartitionSize);
          partitionCount = TPCCConstants.getGlobalPartitionIds().size();
        }
      }

      for (int i = 0; i < partitionCount; i++) {
        this.partitions.add(new PartitionInfo(i));
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

  public void setIsolation(int partitionId, IsolationLevelType isolationLevel) {
    this.partitions.get(partitionId).setLevel(isolationLevel);
  }

  public void setMu(int partitionId, int mu) {
    this.partitions.get(partitionId).setMu(mu);
  }

  public IsolationLevelType getIsolation(int partitionId) {
    if (partitionId < 0 || partitionId >= this.partitions.size()) {
      logger.error("Partition id out of bounds: " + partitionId);
    }

    return this.partitions.get(partitionId).getLevel();
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
          partitionConfig.getIsolationLevel(
              (System.currentTimeMillis() - startTime) / 1000, partitionId);
    } else {
      isolationLevelType = partitions.get(partitionId).getLevel();
    }

    switch (isolationLevelType) {
      case RC -> {
        return 0;
      }
      case SI -> {
        return 1;
      }
      case SER -> {
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
      return partitions.get(partitionId).getMu();
    }
  }

  /*
   * @return global unique partition id
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
    String relationName = validationMeta.getRelationName();
    int key = validationMeta.getIdForValidation();
    return getPartitionId(relationName, key);
  }

  public int getPartitionId(String relationName, int key) {
    return switch (workload) {
      case "ycsb" -> {
        if (!YCSBConstants.TABLENAME_TO_HASH_SIZE.containsKey(relationName)) {
          logger.error("Unknown relation name: {}", relationName);
          yield -1;
        }
        yield YCSBConstants.getGlobalPartitionIdByKey(relationName, key);
      }
      case "smallbank" -> {
        if (!SmallBankConstants.TABLENAME_TO_HASH_SIZE.containsKey(relationName)) {
          logger.error("Unknown relation name: {}", relationName);
          yield -1;
        }
        yield SmallBankConstants.getGlobalPartitionIdByKey(relationName, key);
      }
      case "tpcc" -> {
        if (!TPCCConstants.TABLENAME_TO_HASH_SIZE.containsKey(relationName)) {
          logger.error("Unknown relation name: {}", relationName);
          yield -1;
        }
        yield TPCCConstants.getGlobalPartitionIdByKey(relationName, key);
      }
      default -> {
        logger.error("Unknown workload: {}", workload);
        yield -1;
      }
    };
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

  public DataItem getAndAddDataItem(ValidationMeta validationMeta, long timestamp) {
    int bucketNum =
        validationMeta.getIdForValidation()
            % getHashSizeByRelationName(validationMeta.getRelationName());
    ReadWriteLock lock = tableToDataItemGuards.get(validationMeta.getRelationName()).get(bucketNum);
    lock.readLock().lock();
    for (DataItem item : tableToDataItems.get((validationMeta.getRelationName())).get(bucketNum)) {
      if (item.getKey() == validationMeta.getIdForValidation()) {
        item.clearVersions(timestamp);
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

    public PartitionInfo(int partitionId) {
      this.partitionId = partitionId;
      this.mu = 2;
      this.level = IsolationLevelType.SER;
    }
  }
}
