package org.dbiir.txnsails.execution.validation;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import lombok.Getter;
import org.dbiir.txnsails.common.types.CCType;
import org.dbiir.txnsails.common.types.LockType;

public class ValidationLock {
  private static final long EXPIRED_SPAN = 1000; // the expired span is 1s
  private LockType type; // SH for read validation, EX for write commit check
  private final Lock lock;
  private int count;
  private long maxTid;
  private long minWriteWaitTid;
  private long minReadWaitTid;
  @Getter private final long id;
  @Getter long version;
  private long lease;

  public ValidationLock(long id) {
    this.lock = new ReentrantLock();
    this.type = LockType.NoneType;
    this.count = 0;
    this.maxTid = 0;
    this.minWriteWaitTid = 0;
    this.minReadWaitTid = 0;
    this.id = id;
    this.version = -1;
    this.lease = System.currentTimeMillis() + EXPIRED_SPAN;
  }

  public synchronized void updateLease() {
    lease = System.currentTimeMillis() + EXPIRED_SPAN;
  }

  public synchronized boolean isExpired() {
    return System.currentTimeMillis() > lease;
  }

  public synchronized void updateVersion(long v) {
    version = v > version ? v : version;
  }

  /*
   * @return result. 1 for success, 0 means the transaction manager can wait, -1 for abort
   */
  public int tryLock(long tid, LockType lockType, CCType ccType) {
    int result = -2;
    this.lock.lock();
    if (this.type == LockType.NoneType) {
      this.type = lockType;
      assert (this.count == 0);
      count++;
      if (lockType == LockType.EX) {
        minWriteWaitTid = 0;
      } else if (lockType == LockType.SH) {
        minReadWaitTid = 0;
      }
      this.maxTid = tid;
      result = 1;
    } else if (this.type == LockType.SH) {
      // this type is SH
      if (lockType == LockType.SH) {
        if (minWriteWaitTid != 0 && tid > minWriteWaitTid) {
          // an old transaction wants to read the entry, abort
          result = -1;
        } else {
          count++;
          this.maxTid = Math.max(this.maxTid, tid);
          result = 1;
        }
      } else {
        if (minWriteWaitTid != 0) {
          // it wants to acquire a write lock, while there is a concurrent write send commit before
          // it
          if (ccType == CCType.SI_TAILOR) {
            result = -1;
          } else if (ccType == CCType.RC_TAILOR) {
            minWriteWaitTid = Math.min(minWriteWaitTid, tid);
            result = 0;
          }
        } else {
          minWriteWaitTid = tid;
          result = 0;
        }

        if (result == 0 && maxTid > tid) {
          result = -1;
        }
      }
    } else {
      // this type is EX
      if (lockType == LockType.SH) {
        if (ccType == CCType.RC_TAILOR) {
          if (minReadWaitTid == 0) minReadWaitTid = tid;
          else minReadWaitTid = Math.min(tid, minReadWaitTid);
        }
        result = -1;
      } else {
        if (ccType == CCType.SI_TAILOR) {
          result = -1;
        } else if (ccType == CCType.RC_TAILOR) {
          // concurrent update is allowed under RC, only need to keep the rw order
          count++;
          this.maxTid = Math.max(this.maxTid, tid);
          result = 1;
          // if (minReadWaitTid != 0 && tid > minReadWaitTid) {
          //     // System.out.println("Transaction #"+tid + " rollback, minReadWait: " +
          // minReadWaitTid);
          //     // result = -1;
          //     minWriteWaitTid = minWriteWaitTid == 0 ? tid : Math.min(minWriteWaitTid, tid);
          //     result = 0;
          // } else {
          //     count++;
          //     this.maxTid = Math.max(this.maxTid, tid);
          //     result = 1;
          // }
        }
      }
    }
    this.lock.unlock();
    assert (result != -2);
    return result;
  }

  public void releaseLock(LockType lockType) {
    this.lock.lock();
    assert (this.type == lockType);
    count--;
    if (count == 0) {
      this.type = LockType.NoneType;
      this.maxTid = 0;
    }
    this.lock.unlock();
  }

  public boolean free() {
    return this.type == LockType.NoneType;
  }
}
