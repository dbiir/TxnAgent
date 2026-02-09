package org.dbiir.txnagent.common.types;

public enum LockStrategyType {
  NO_WAIT("No wait"),
  WAIT_DIE("Wait die");

  private final String description;

  LockStrategyType(String description) {
    this.description = description;
  }

  @Override
  public String toString() {
    return description;
  }
}
