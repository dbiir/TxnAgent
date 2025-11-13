package org.dbiir.txnsails.execution.isolation;

import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnsails.common.constants.SmallBankConstants;
import org.dbiir.txnsails.common.constants.TPCCConstants;
import org.dbiir.txnsails.common.constants.YCSBConstants;
import org.dbiir.txnsails.common.types.CCType;
import org.dbiir.txnsails.execution.validation.ValidationLock;
import org.dbiir.txnsails.execution.validation.ValidationMeta;
import org.dbiir.txnsails.execution.validation.ValidationMetaTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class PartitionManager {
  private static Logger logger = LoggerFactory.getLogger(PartitionManager.class);
  private static final PartitionManager INSTANCE;
  private final HashMap<String, ArrayList<LinkedList<DataItem>>> tableToDataItems;
  private final HashMap<String, ArrayList<ReentrantReadWriteLock>> tableToDataItemGuards;
  private Thread stasticThread;
  @Getter private String workload;

  static {
    INSTANCE = new PartitionManager();
  }

  public PartitionManager() {
    this.tableToDataItems = new HashMap<>();
    this.tableToDataItemGuards = new HashMap<>();
    // create the statistic thread
    this.stasticThread = new Thread(new StasticThread());
    this.stasticThread.start();
  }

  public void init(String workload) {
    // TODO:
    this.workload = workload;
    switch (workload) {
      case "ycsb":
        for (Map.Entry<String, Integer> entry: YCSBConstants.TABLENAME_TO_HASH_SIZE.entrySet()) {
          if (entry.getValue() <= 0) {
            continue;
          }
          createItemTable(entry.getKey(), entry.getValue());
        }
        break;
      case "smallbank":
        for (Map.Entry<String, Integer> entry: SmallBankConstants.TABLENAME_TO_HASH_SIZE.entrySet()) {
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
      dataItems.add(new DataItem(i));
      tableToDataItems.get(relationName).add(dataItems);
    }
  }

  public int chooseIsolation(ValidationMeta validationMeta) {
    return 0;
  }

  public DataItem getDataItem(ValidationMeta validationMeta) {
    int bucketNum = (int) (validationMeta.getIdForValidation() % getHashSizeByRelationName(validationMeta.getRelationName()));
    ReadWriteLock lock = tableToDataItemGuards.get(validationMeta.getRelationName()).get(bucketNum);
    lock.readLock().lock();
    for (DataItem item: tableToDataItems.get((validationMeta.getRelationName())).get(bucketNum)) {
      if (item.getKey() == validationMeta.getIdForValidation()) {
        lock.readLock().unlock();
        return item;
      }
    }
    lock.readLock().unlock();
    return null;
  }

  public DataItem getAndAddDataItem(ValidationMeta validationMeta) {
    int bucketNum = (int) (validationMeta.getIdForValidation() % getHashSizeByRelationName(validationMeta.getRelationName()));
    ReadWriteLock lock = tableToDataItemGuards.get(validationMeta.getRelationName()).get(bucketNum);
    lock.readLock().lock();
    for (DataItem item: tableToDataItems.get((validationMeta.getRelationName())).get(bucketNum)) {
      if (item.getKey() == validationMeta.getIdForValidation()) {
        lock.readLock().unlock();
        return item;
      }
    }
    lock.readLock().unlock();
    lock.writeLock().lock();
    DataItem item = new DataItem(validationMeta.getIdForValidation());
    tableToDataItems.get((validationMeta.getRelationName())).get(bucketNum).add(item);
    lock.writeLock().unlock();
    return item;
  }

  // TODO:
  public void collect(Transaction transaction) {
    for (Participant participant : transaction.getParticipants()) {

    }
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

  public void close() {
    this.stasticThread.interrupt();
  }

  public static PartitionManager getInstance() {
    return INSTANCE;
  }

  private class StasticThread extends Thread {
    private static final String ip = "localhost";
    private static final int port = 7654;
    @Setter
    private String outputFilePrefix;
    private CCType ccType;
    private boolean online;
    private Socket socket;

    @Override
    public void run() {
      while (!Thread.interrupted()) {
        try {
          // TODO: collect the statistics of data items
          Thread.sleep(5000L);
        } catch (InterruptedException e) {
          logger.info("Stastic thread interrupted: {}", e.getMessage());
        }
      }
    }
  }
}
