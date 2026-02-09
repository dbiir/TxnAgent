package org.dbiir.txnagent.common;

public enum TransactionStatus {
  IDLE("Idle"), // none transaction
  ACTIVE("Active"), // in processing transaction
  PREPARED("Prepared"),
  VALIDATED("Validated"),
  PREPARE_FAILED("PrepareFailed"),
  COMMITTED("Committed"),
  ROLLBACK("Rollback");

  private final String description;

  TransactionStatus(String description) {
    this.description = description;
  }

  @Override
  public String toString() {
    return description;
  }
}
