package org.dbiir.txnagent.common.types;

public enum DependencyType {
  READ_WRITE("RW Dependency"),
  WRITE_READ("WR Dependency"),
  WRITE_WRITE("WW Dependency");

  private final String description;

  DependencyType(String description) {
    this.description = description;
  }

  @Override
  public String toString() {
    return description;
  }
}
