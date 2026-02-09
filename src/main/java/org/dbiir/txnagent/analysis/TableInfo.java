package org.dbiir.txnagent.analysis;

import java.util.HashMap;
import lombok.Getter;
import org.dbiir.txnagent.common.types.ColumnType;

@Getter
public class TableInfo {
  private final String tableName;
  private final HashMap<String, ColumnInfo> columns;

  public TableInfo(String tableName) {
    this.tableName = tableName;
    this.columns = new HashMap<>();
  }

  public void addColumn(ColumnInfo column) {
    columns.put(column.getName().toUpperCase(), column);
  }

  public ColumnInfo findColumn(String colName) {
    if (columns.containsKey(colName.toUpperCase())) {
      return columns.get(colName.toUpperCase());
    }
    return null;
  }

  public ColumnType findColumnType(String colName) {
    if (columns.containsKey(colName.toUpperCase())) {
      return columns.get(colName.toUpperCase()).getType();
    }
    return null;
  }

  @Override
  public String toString() {
    return "TableSchema{" + "tableName='" + tableName + '\'' + ", columns=" + columns + '}';
  }
}
