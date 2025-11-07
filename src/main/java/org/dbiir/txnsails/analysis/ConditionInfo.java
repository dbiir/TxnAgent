package org.dbiir.txnsails.analysis;

import lombok.Getter;
import net.sf.jsqlparser.expression.JdbcParameter;
import net.sf.jsqlparser.schema.Column;

public class ConditionInfo {
  private final Column column;
  @Getter private final String upperCaseColumnName;
  private final JdbcParameter parameter;

  public ConditionInfo(Column column, JdbcParameter parameter) {
    this.column = column;
    this.upperCaseColumnName = column.getColumnName().toUpperCase();
    this.parameter = parameter;
  }

  public String getColumnName() {
    return column.getColumnName();
  }

  public String getColumnFullName() {
    return column.getFullyQualifiedName();
  }

  public int getPlaceholderIndex() {
    return parameter.getIndex() - 1;
  }
}
