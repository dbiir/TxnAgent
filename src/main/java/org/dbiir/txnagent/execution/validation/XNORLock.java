package org.dbiir.txnagent.execution.validation;

import java.sql.SQLException;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import lombok.Getter;
import org.dbiir.txnagent.common.types.LockStrategyType;
import org.dbiir.txnagent.common.types.LockType;

public class XNORLock {
  private LockType type;
  private final List<LockEntry> waitList;
  // for debug
  private final List<LockEntry> lockList;
  private int count;
  private final Lock lock;
  private final LockStrategyType strategy; // 1 for no-wait, 2 for wait-die
  @Getter private final long key;
  private static long lockWaitTimeout = 10;

  public XNORLock() {
    this.type = LockType.NoneType;
    this.waitList = new LinkedList<>();
    this.lockList = new LinkedList<>();
    this.count = 0;
    this.key = 0;
    lock = new ReentrantLock();
    strategy = LockStrategyType.NO_WAIT;
  }

  public XNORLock(long key) {
    this.type = LockType.NoneType;
    this.waitList = new LinkedList<>();
    this.lockList = new LinkedList<>();
    this.count = 0;
    this.key = key;
    lock = new ReentrantLock();
    strategy = LockStrategyType.NO_WAIT;
  }

  public void tryLock(long transactionId, LockType type) throws SQLException {
    lock.lock();
    try {
      if (!waitList.isEmpty() || !tolerate(this.type, type)) {
        if (strategy == LockStrategyType.NO_WAIT) {
          lock.unlock();
          String msg = "Abort due to no wait, transaction id: " + transactionId;
          throw new SQLException(msg, "500");
        }
        if (!admitByWaitDieStrategy(transactionId)) {
          lock.unlock();
          String msg = "Abort due to wait die, transaction id: " + transactionId;
          throw new SQLException(msg, "500");
        }
        LockEntry entry = new LockEntry(transactionId, type);
        waitList.add(entry);
        lock.unlock();
        long startTime = System.currentTimeMillis();
        while (!entry.isGrantee()) {
          Thread.sleep(0, 1000);
          if (System.currentTimeMillis() - startTime > lockWaitTimeout) {
            lock.lock();
            waitList.removeIf(e -> e.getTransactionId() == transactionId);
            lock.unlock();
            String msg = "Abort due to wait die and timeout, transaction id: " + transactionId;
            throw new SQLException(msg, "500");
          }
        }
      } else {
        this.type = type;
        lockList.add(new LockEntry(transactionId, type, true));
        lock.unlock();
      }
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
    }
  }

  public boolean releaseLock(long tid) {
    lock.lock();
    try {
      lockList.removeIf(entry -> entry.getTransactionId() == tid);
      if (!lockList.isEmpty()) return true;

      this.type = LockType.NoneType;
      if (!waitList.isEmpty()) {
        Iterator<LockEntry> waitListIterator = waitList.iterator();
        while (waitListIterator.hasNext()) {
          LockEntry llock = waitListIterator.next();
          if (tolerate(this.type, llock.getLockType())) {
            lockList.add(llock);
            waitListIterator.remove();
            llock.setGrantee(true);
          } else {
            break;
          }
        }
      }
      return true;
    } finally {
      lock.unlock();
    }
  }

  private boolean tolerate(LockType type1, LockType type2) {
    if (type1 == LockType.NoneType || type2 == LockType.NoneType) return true;
    //        if (type1 == LockType.EX || type2 == LockType.EX)
    //            return false;
    return type1 == type2;
  }

  private boolean admitByWaitDieStrategy(long tid) {
    for (LockEntry entry : lockList) {
      assert (tid != entry.getTransactionId());
      if (tid > entry.getTransactionId()) return false;
    }

    for (LockEntry entry : waitList) {
      assert (tid != entry.getTransactionId());
      if (tid > entry.getTransactionId()) return false;
    }
    return true;
  }

  public boolean free() {
    if (lock.tryLock()) return false;
    if (lockList.isEmpty() && waitList.isEmpty()) {
      lock.unlock();
      return true;
    } else {
      return false;
    }
  }

  public String toString() {
    StringBuilder str = new StringBuilder();
    str.append("{type: { ").append(type).append(" }, \nlock list: { \n");
    for (LockEntry entry : lockList) {
      str.append(entry);
    }
    str.append(" }, wait list: { \n");
    for (LockEntry entry : waitList) {
      str.append(entry);
    }
    str.append(" }}");
    return str.toString();
  }
}
