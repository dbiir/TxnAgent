package org.dbiir.txnagent.common;

import lombok.Getter;
import org.dbiir.txnagent.common.types.CCType;

public class TransactionTemplate implements Cloneable {
  @Getter private final String name;
  // a template contains at most 15 sqls for now
  private final TemplateSQL[] sqls = new TemplateSQL[15];
  private int sql_idx_in_template;
  @Getter private int type; // one-hot encode

  public TransactionTemplate(String name) {
    this.name = name;
    this.sql_idx_in_template = 0;
    this.type = -1;
  }

  public TransactionTemplate(String name, int type) {
    this.name = name;
    this.sql_idx_in_template = 0;
    this.type = type;
  }

  public int addTemplateSQL(int op, String relation, String sql) {
    this.sqls[sql_idx_in_template] = new TemplateSQL(op, relation, sql);
    sql_idx_in_template++;
    return sql_idx_in_template - 1;
  }

  // return the rewrite sql
  public String getSQLByIndex(int idx, CCType ccType) {
    if (idx >= sql_idx_in_template) {
      System.out.println("template SQL's index out of bound");
      return "";
    }
    return sqls[idx].getSQL();
  }

  // return the sql template
  public TemplateSQL getSQLTemplateByIndex(int idx) {
    if (idx >= sql_idx_in_template) {
      System.out.println("template SQL's index out of bound");
      return null;
    }
    return sqls[idx];
  }

  public int totalSQLs() {
    return sql_idx_in_template;
  }

  public int getOperationByIndex(int idx) {
    return sqls[idx].getOp();
  }

  public String getRelationByIndex(int idx) {
    return sqls[idx].getTable();
  }

  public void setSQLRewriteByIndex(int idx, String isolation) {
    if (isolation.equals("RC")) {
      sqls[idx].setNeedRewriteUnderRC();
    } else if (isolation.equals("SI")) {
      sqls[idx].setNeedRewriteUnderSI();
    }
  }

  public boolean shouldSkipIndex(int idx) {
    return sqls[idx].isSkip();
  }

  public void setSkipIndex(int idx, boolean skip) {
    sqls[idx].setSkip(skip);
  }

  @Override
  public TransactionTemplate clone() {
    try {
      TransactionTemplate clone = (TransactionTemplate) super.clone();
      // TODO: copy mutable state here, so the clone can't change the internals of the original
      //            clone.sqls = new TemplateSQL[15];
      for (int i = 0; i < sql_idx_in_template; i++) {
        if (sqls[i] != null) {
          clone.sqls[i] = sqls[i].clone();
        }
      }
      return clone;
    } catch (CloneNotSupportedException e) {
      System.out.println(e.toString());
      throw new AssertionError();
    }
  }
}
