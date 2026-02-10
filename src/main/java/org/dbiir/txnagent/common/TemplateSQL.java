package org.dbiir.txnagent.common;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import lombok.Getter;
import lombok.Setter;
import net.sf.jsqlparser.JSQLParserException;
import net.sf.jsqlparser.expression.*;
import net.sf.jsqlparser.expression.operators.arithmetic.Addition;
import net.sf.jsqlparser.expression.operators.conditional.AndExpression;
import net.sf.jsqlparser.expression.operators.conditional.OrExpression;
import net.sf.jsqlparser.expression.operators.relational.*;
import net.sf.jsqlparser.parser.*;
import net.sf.jsqlparser.schema.*;
import net.sf.jsqlparser.statement.*;
import net.sf.jsqlparser.statement.select.*;
import net.sf.jsqlparser.statement.update.*;
import org.dbiir.txnagent.analysis.ColumnInfo;
import org.dbiir.txnagent.analysis.ConditionInfo;
import org.dbiir.txnagent.common.constants.YCSBConstants;
import org.dbiir.txnagent.worker.MetaWorker;

@Getter
public class TemplateSQL implements Cloneable {
  private final int op; // 0 for read, 1 for write, 2 for scan
  private final String originSQL;
  private String rewriteSQL;
  private final String table; // join operation in future
  @Setter private boolean skip;
  private boolean needRewriteUnderSI;
  private boolean needRewriteUnderRC;
  @Setter private List<Integer> uniqueKeyIndexList; // calculate
  private int uniqueKeyNumber;
  @Setter @Getter private List<Column> columnList;
  @Getter private List<ConditionInfo> wherePlaceholders;
  @Getter private List<ConditionInfo> allPlaceholders;
  private boolean selectAllAttr;

  public TemplateSQL(int op, String table, String sql) {
    this.op = op;
    this.originSQL = sql;
    this.table = table;
    this.columnList = new ArrayList<>(4);
    this.needRewriteUnderSI = false;
    this.needRewriteUnderRC = false;
    this.skip = false;
    this.uniqueKeyIndexList = new ArrayList<>(4);
    this.rewriteSQL = "";
    this.wherePlaceholders = new ArrayList<>();
    this.allPlaceholders = new ArrayList<>();
    this.selectAllAttr = false;

    // find the placeholders in the clause
    findJdbcParameters();
    // fill the column list
    fillColumnList();
  }

  public void addUniqueKeyIndex(int idx) {
    this.uniqueKeyIndexList.add(idx);
    this.uniqueKeyNumber++;
  }

  public void setNeedRewriteUnderRC() {
    this.needRewriteUnderRC = true;
    rewrite();
  }

  public void setNeedRewriteUnderSI() {
    this.needRewriteUnderSI = true;
    rewrite();
  }

  private void rewrite() {
    if (!rewriteSQL.isEmpty()) return;
    if (identifySQLType(originSQL) == 1) {
      this.rewriteSQL = modifyUpdateQuery();
      System.out.println("1 rewriteSQL: " + rewriteSQL);
    } else if (identifySQLType(originSQL) == 0) {
      this.rewriteSQL = modifySelectQuery();
      System.out.println("2 rewriteSQL: " + rewriteSQL);
    } else {
      this.rewriteSQL = "";
      System.out.println("3 rewriteSQL: " + rewriteSQL);
    }
    System.out.println("rewriteSQL: " + rewriteSQL);
  }

  public int getUniqueKeyIdx(int idx) {
    return this.uniqueKeyIndexList.get(idx);
  }

  public String getSQL() {
//    if (needRewriteUnderRC || needRewriteUnderSI) {
//      return rewriteSQL;
//    }
//    return originSQL;
    return rewriteSQL;
  }

  private String getSQLUnderRC() {
    if (needRewriteUnderRC) {
      return rewriteSQL;
    } else {
      return originSQL;
    }
  }

  private String getSQLUnderSI() {
    if (needRewriteUnderSI) {
      return rewriteSQL;
    } else {
      return originSQL;
    }
  }

  private void fillColumnList() {
    CCJSqlParserManager parserManager = new CCJSqlParserManager();
    try {
      if (op == 0) {
        // read
        Select selectStatement = (Select) parserManager.parse(new StringReader(originSQL));
        PlainSelect plainSelect = (PlainSelect) selectStatement.getSelectBody();
        parseSelectItems(plainSelect.getSelectItems());
      } else {
        // write, parse the returning clause
        Update updateStatement = (Update) parserManager.parse(new StringReader(originSQL));
        parseSelectItems(updateStatement.getReturningExpressionList());
      }
    } catch (Exception e) {
      System.out.println(e.toString());
      System.out.println(
          Arrays.stream(e.getStackTrace())
              .map(StackTraceElement::toString)
              .collect(Collectors.joining("\n")));
    }
  }

  private void parseSelectItems(List<SelectItem> selectItemList) {
    if (selectItemList == null) {
      return;
    }

    for (SelectItem selectItem : selectItemList) {
      if (selectItem instanceof SelectExpressionItem selectExpressionItem) {
        if (selectExpressionItem.getExpression() instanceof Column column) {
          columnList.add(column);
        }
      } else if (selectItem instanceof AllColumns) {
        selectAllAttr = true;
        for (ColumnInfo c : MetaWorker.getINSTANCE().getSchema().getColumnInfoByTableName(table)) {
          columnList.add(new Column(c.getName()));
        }
      } else {
        System.out.println("unsupported select expression: " + selectItem);
      }
    }
  }

  // Method: Add the field "vid" to the SELECT statement and ensure it is added at the end
  private String modifySelectQuery() {
    try {
      // Parse the original SELECT statement
      CCJSqlParserManager parserManager = new CCJSqlParserManager();
      Select selectStatement = (Select) parserManager.parse(new StringReader(originSQL));

      // Get the SELECT body, which is the part containing the fields
      PlainSelect plainSelect = (PlainSelect) selectStatement.getSelectBody();

      // Get the list of SELECT items (fields)
      List<SelectItem> selectItems = plainSelect.getSelectItems();

      if (!selectAllAttr) {
        // Create a new column "vid"
        Column vidColumn = new Column("vid");

        // Create a new SELECT expression item (field)
        SelectExpressionItem vidSelectItem = new SelectExpressionItem(vidColumn);

        // Add the "vid" field to the SELECT clause and Column list
        selectItems.add(vidSelectItem);
        columnList.add(vidColumn);
      }

      // Return the modified SQL statement
      System.out.println("selectStatement: " + selectStatement.toString());
      return selectStatement.toString();

    } catch (Exception e) {
      System.out.println(e.toString());
      System.out.println(
          Arrays.stream(e.getStackTrace())
              .map(StackTraceElement::toString)
              .collect(Collectors.joining("\n")));
      return null;
    }
  }

  // Method: Modify the SQL statement to add the "vid" field and return it
  private String modifyUpdateQuery() {
    try {
      // Parse the SQL statement
      CCJSqlParserManager parserManager = new CCJSqlParserManager();
      Update updateStatement = (Update) parserManager.parse(new StringReader(originSQL));

      // Create the expression "vid = vid + 1"
      Column vidLeftColumn = new Column("vid");
      LongValue oneValue = new LongValue(1); // Represents the constant 1
      Addition addExpression = new Addition(); // Addition represents the "vid + 1" operation
      Column vidRightColumn = new Column("vid");
      if (updateStatement.getFromItem() != null
          && updateStatement.getFromItem().getAlias() != null
          && updateStatement.getFromItem().getAlias().isUseAs()
          && updateStatement.getFromItem() instanceof Table) {
        vidRightColumn.setTable((Table) updateStatement.getFromItem());
      }
      addExpression.setLeftExpression(vidRightColumn); // Left side is the "vid" column
      addExpression.setRightExpression(oneValue); // Right side is the constant value 1

      // Add the SetExpression to the UPDATE statement
      updateStatement
          .getUpdateSets()
          .add(
              new UpdateSet(vidLeftColumn, addExpression)); // Add "vid = vid + 1" to the SET clause

      // Add `vid` to the RETURNING clause
      if (!selectAllAttr) {
        Column vidReturnColumn = new Column("vid");
        if (updateStatement.getFromItem() != null
            && updateStatement.getFromItem().getAlias() != null
            && updateStatement.getFromItem().getAlias().isUseAs()
            && updateStatement.getFromItem() instanceof Table) {
          vidReturnColumn.setTable((Table) updateStatement.getFromItem());
        }
        if (updateStatement.getReturningExpressionList() == null) {
          updateStatement.setReturningExpressionList(
              List.of(new SelectExpressionItem(vidReturnColumn)));
        } else {
          updateStatement
              .getReturningExpressionList()
              .add(new SelectExpressionItem(vidReturnColumn));
        }
        columnList.add(new Column("vid"));
      }

      // Return the modified SQL statement
      System.out.println("updateStatement: " + updateStatement.toString());
      return updateStatement.toString();

    } catch (Exception e) {
      System.out.println(e.toString());
      System.out.println(
          Arrays.stream(e.getStackTrace())
              .map(StackTraceElement::toString)
              .collect(Collectors.joining("\n")));
      return null;
    }
  }

  // Method: Identify the type of SQL statement (SELECT, UPDATE, etc.)
  // 0 for SELECT, 1 for UPDATE
  private int identifySQLType(String sql) {
    try {
      // Create the SQL parser
      CCJSqlParserManager parserManager = new CCJSqlParserManager();

      // Parse the SQL statement
      Statement statement = parserManager.parse(new StringReader(sql));

      // Check the type of SQL statement
      if (statement instanceof Select) {
        return 0;
      } else if (statement instanceof Update) {
        return 1;
      } else {
        return -1;
      }

    } catch (Exception e) {
      e.printStackTrace();
      return -1;
    }
  }

  private void findWherePlaceholders() {
    try {
      Statement statement = CCJSqlParserUtil.parse(originSQL);
      if (statement instanceof Update updateStatement) {
        Expression where = updateStatement.getWhere();
        findConditions(where);
      } else if (statement instanceof Select selectStatement) {
        PlainSelect plainSelect = (PlainSelect) selectStatement.getSelectBody();
        Expression where = plainSelect.getWhere();
        findConditions(where);
      }
    } catch (JSQLParserException e) {
      e.printStackTrace();
    }
  }

  private void findConditions(Expression expression) {
    if (expression instanceof AndExpression) {
      AndExpression andExpression = (AndExpression) expression;
      findConditions(andExpression.getLeftExpression());
      findConditions(andExpression.getRightExpression());
    } else if (expression instanceof OrExpression) {
      OrExpression orExpression = (OrExpression) expression;
      findConditions(orExpression.getLeftExpression());
      findConditions(orExpression.getRightExpression());
    } else if (expression instanceof EqualsTo
        || expression instanceof GreaterThan
        || expression instanceof MinorThan
        || expression instanceof GreaterThanEquals
        || expression instanceof MinorThanEquals) {
      BinaryExpression binaryExpression = (BinaryExpression) expression;
      Column column = (Column) binaryExpression.getLeftExpression();
      Expression rightExpression = binaryExpression.getRightExpression();
      String value = null;
      int position = -1;
      if (rightExpression instanceof JdbcParameter) {
        System.out.println(
            column.getColumnName() + ": " + ((JdbcParameter) rightExpression).getIndex());
        this.wherePlaceholders.add(new ConditionInfo(column, (JdbcParameter) rightExpression));
      } else {
        System.out.println(
            "[No placeholder]: " + column.getColumnName() + ", " + (StringValue) rightExpression);
      }
    }
  }

  // specific for YCSB workload
  private void analyseTemplate() {
    if (this.table.equals(YCSBConstants.TABLE_NAME)) {
      this.needRewriteUnderRC = true;
      this.needRewriteUnderSI = true;
      rewrite();
    }
  }

  private void findJdbcParameters() {
    try {
      Statement statement = CCJSqlParserUtil.parse(originSQL);
      analyseTemplate();

      if (statement instanceof Select select) {
        PlainSelect plainSelect = (PlainSelect) select.getSelectBody();
        List<SelectItem> selectItems = plainSelect.getSelectItems();

        for (SelectItem item : selectItems) {
          if (item instanceof SelectExpressionItem selectExpressionItem) {
            if (selectExpressionItem.getExpression() instanceof JdbcParameter) {
              System.out.println("Not support jdbc parameters in selected attributes");
            }
          }
        }

        // Check WHERE clause
        Expression where = plainSelect.getWhere();
        if (where != null) {
          findJdbcParametersInWhere(where);
        }

      } else if (statement instanceof Update update) {
        List<UpdateSet> updateSets = update.getUpdateSets();

        for (UpdateSet updateSet : updateSets) {
          List<Column> columns = updateSet.getColumns();
          for (Column column : columns) {
            if (updateSet.getExpressions().get(columns.indexOf(column))
                instanceof JdbcParameter param1) {
              allPlaceholders.add(new ConditionInfo(column, param1));
            }
          }
          for (Expression expression : updateSet.getExpressions()) {
            findJdbcParametersInSet(expression);
          }
        }

        // Check WHERE clause
        Expression where = update.getWhere();
        if (where != null) {
          findJdbcParametersInWhere(where);
        }
      }
    } catch (Exception e) {
      System.out.println("originSQL: " + originSQL);
      e.printStackTrace();
    }
  }

  private void findJdbcParametersInSet(Expression where) {
    if (where instanceof BinaryExpression binaryExpression) {
      Expression leftExpression = binaryExpression.getLeftExpression();
      Expression rightExpression = binaryExpression.getRightExpression();

      if (rightExpression instanceof JdbcParameter param1
          && leftExpression instanceof Column column1) {
        allPlaceholders.add(new ConditionInfo(column1, param1));
      }
      if (rightExpression instanceof Column column2
          && leftExpression instanceof JdbcParameter param2) {
        allPlaceholders.add(new ConditionInfo(column2, param2));
      }

      // Recursively check nested expressions
      findJdbcParametersInSet(leftExpression);
      findJdbcParametersInSet(rightExpression);
    }
  }

  private void findJdbcParametersInWhere(Expression where) {
    if (where instanceof BinaryExpression binaryExpression) {
      Expression leftExpression = binaryExpression.getLeftExpression();
      Expression rightExpression = binaryExpression.getRightExpression();

      if (rightExpression instanceof JdbcParameter param1
          && leftExpression instanceof Column column1) {
        allPlaceholders.add(new ConditionInfo(column1, param1));
        wherePlaceholders.add(new ConditionInfo(column1, param1));
      }
      if (rightExpression instanceof Column column2
          && leftExpression instanceof JdbcParameter param2) {
        allPlaceholders.add(new ConditionInfo(column2, param2));
        wherePlaceholders.add(new ConditionInfo(column2, param2));
      }

      // Recursively check nested expressions
      findJdbcParametersInWhere(leftExpression);
      findJdbcParametersInWhere(rightExpression);
    }
  }

  @Override
  public TemplateSQL clone() {
    try {
      TemplateSQL clone = (TemplateSQL) super.clone();
      // TODO: copy mutable state here, so the clone can't change the internals of the original
      return clone;
    } catch (CloneNotSupportedException e) {
      throw new AssertionError();
    }
  }
}
