package org.dbiir.txnagent.execution.isolation;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import lombok.Getter;
import lombok.Setter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// not thread-safe
public class DataItem {
  private static final Logger logger = LoggerFactory.getLogger(DataItem.class);
  private static final int LEASE_TIME = 10000; // milliseconds
  @Getter @Setter private int key;
  private long lease;
  private final List<RecordVersion> versions;
  @Getter private int partitionId;
  @Getter private String relationName;

  @Getter
  private final List<Transaction>
      readTransactions; // transactions that have read this data item and not committed yet

  private final AtomicLong writeTransaction;
  @Getter private long maxReadTimestamp;
  private final ReadWriteLock lock =
      new ReentrantReadWriteLock(); // support atomic modification of readTransactions
  private final ReadWriteLock versionLock = new ReentrantReadWriteLock();

  public DataItem(int key) {
    this.key = key;
    this.lease = System.currentTimeMillis() + LEASE_TIME;
    this.versions = new java.util.LinkedList<>();
    this.readTransactions = new java.util.LinkedList<>();
    this.writeTransaction = new AtomicLong(0); // ? is necessary ?
    this.maxReadTimestamp = 0;
    this.partitionId = -1;
    this.relationName = null;
  }

  public DataItem(int key, int partitionId, String relationName) {
    this(key);
    this.partitionId = partitionId;
    this.relationName = relationName;
  }

  public DataItem(int key, int partitionId, String relationName, long version) {
    this(key);
    this.partitionId = partitionId;
    this.relationName = relationName;
    this.versions.add(new RecordVersion(version, System.nanoTime()));
  }

  public void read(Transaction transaction, long version) {
    logger.debug("data item {}, read transaction {}", key, transaction.getId());
    this.lock.writeLock().lock();
    this.readTransactions.add(transaction);
    this.lock.writeLock().unlock();
    // CC - Execution: if there is a new version, T_i's.UB = x_{m+1}.cts - 1
    versionLock.readLock().lock();
    for (RecordVersion rv : versions) {
      if (rv.version() <= version) {
        continue;
      }
      transaction.setUpperBound(rv.timestamp() - 1);
      break;
    }
    versionLock.readLock().unlock();
    this.lease = System.currentTimeMillis() + LEASE_TIME;
  }

  public void write(Transaction transaction) {
    logger.debug("data item {}, write transaction {}", key, transaction.getId());
    this.lease = System.currentTimeMillis() + LEASE_TIME;
  }

  public void setMaxReadTimestamp(long ts) {
    this.maxReadTimestamp = Math.max(this.maxReadTimestamp, ts);
  }

  public void removeReadTransaction(Transaction transaction) {
    this.lock.writeLock().lock();
    this.readTransactions.remove(transaction);
    this.lock.writeLock().unlock();
  }

  public void acquireReadLock() {
    this.lock.readLock().lock();
  }

  public void releaseReadLock() {
    assert !this.lock.writeLock().tryLock();
    this.lock.readLock().unlock();
  }

  public boolean addVirtualWriteLock(long tid) {
    return this.writeTransaction.compareAndSet(0, tid);
  }

  public void releaseWriteLock(long tid) {
    this.writeTransaction.compareAndSet(tid, 0);
  }

  public boolean canRemove() {
    return System.currentTimeMillis() > lease
        && this.readTransactions.isEmpty()
        && this.writeTransaction.get() == 0;
  }

  public void installVersion(long version, long commitTimestamp, long minActiveTransactionId) {
    this.versionLock.writeLock().lock();
    if (version == 0) {
      version = this.versions.getLast().version() + 1;
    }
    if (this.versions.size() > 8) {
      this.versions.removeIf(rv -> rv.timestamp() < minActiveTransactionId);
    }
    this.versions.add(new RecordVersion(version, commitTimestamp));
    this.versionLock.writeLock().unlock();
  }
}
