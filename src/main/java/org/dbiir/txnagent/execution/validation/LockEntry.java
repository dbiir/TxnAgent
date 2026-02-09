package org.dbiir.txnagent.execution.validation;

import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.common.types.LockType;

@Getter
@Setter
public class LockEntry {
  private final long transactionId;
  private final LockType lockType;
  private boolean grantee;
  private final long enterTime;

  public LockEntry(long transactionId, LockType type) {
    this.transactionId = transactionId;
    this.lockType = type;
    this.enterTime = System.currentTimeMillis();
    this.grantee = false;
  }

  public LockEntry(long transactionId, LockType type, boolean grantee) {
    this.transactionId = transactionId;
    this.lockType = type;
    this.enterTime = System.currentTimeMillis();
    this.grantee = grantee;
  }

  public boolean Timeout() {
    return (System.nanoTime() - enterTime) / 1000000 > 5000;
  }

  public String toString() {
    return "{transaction id: { "
        + transactionId
        + " }, type: { "
        + lockType
        + " }, enter time: { "
        + enterTime
        + "}}";
  }
}
