package org.dbiir.txnagent.analysis;

import lombok.Getter;
import org.dbiir.txnagent.common.types.ColumnType;

@Getter
public class ColumnInfo {
  // Getters and toString() method
  private final String name;
  private final ColumnType type;
  private final boolean isPrimaryKey;

  public ColumnInfo(String name, String type, boolean isPrimaryKey) {
    this.name = name;
    this.type = convertColumnTypeFromString(type);
    this.isPrimaryKey = isPrimaryKey;
  }

  private ColumnType convertColumnTypeFromString(String columnType) {
    return switch (columnType.toUpperCase()) {
      case "INTEGER", "INT" -> ColumnType.INTEGER;
      case "BIGINT" -> ColumnType.BIGINT;
      case "FLOAT" -> ColumnType.FLOAT;
      case "DOUBLE", "DECIMAL" -> ColumnType.DOUBLE;
      case "BOOLEAN" -> ColumnType.BOOLEAN;
      case "VARCHAR" -> ColumnType.VARCHAR;
      case "TEXT" -> ColumnType.TEXT;
      case "DATE" -> ColumnType.DATE;
      case "TIMESTAMP" -> ColumnType.TIMESTAMP;
      default -> throw new IllegalArgumentException("Unsupported column type");
    };
  }

  @Override
  public String toString() {
    return "Column{"
        + "name='"
        + name
        + '\''
        + ", type='"
        + type
        + '\''
        + ", isPrimaryKey="
        + isPrimaryKey
        + '}';
  }
}
