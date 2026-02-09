package org.dbiir.txnagent.common.types;

public enum ColumnType {
  INTEGER("Integer"),
  BIGINT("Big int"),
  FLOAT("Float"),
  DOUBLE("Double"),
  BOOLEAN("boolean"),
  VARCHAR("Text string"),
  TEXT("Text string"),
  DATE("Date"),
  TIMESTAMP("Timestamp");

  private final String description;

  ColumnType(String description) {
    this.description = description;
  }

  @Override
  public String toString() {
    return description;
  }
}
