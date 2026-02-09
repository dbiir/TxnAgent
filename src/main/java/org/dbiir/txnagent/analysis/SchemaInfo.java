package org.dbiir.txnagent.analysis;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import net.sf.jsqlparser.parser.CCJSqlParserUtil;
import net.sf.jsqlparser.statement.Statement;
import net.sf.jsqlparser.statement.create.table.ColDataType;
import net.sf.jsqlparser.statement.create.table.ColumnDefinition;
import net.sf.jsqlparser.statement.create.table.CreateTable;
import net.sf.jsqlparser.statement.drop.Drop;
import org.dbiir.txnagent.common.types.ColumnType;

public class SchemaInfo {
  private final List<TableInfo> tables;

  public SchemaInfo() {
    this.tables = new ArrayList<>();
  }

  public SchemaInfo(String fileName) {
    this.tables = new ArrayList<>();
    processSQLStatements(readSQLFromFile(fileName));
    //      extractSchemaFromFile(fileName);
  }

  private String readSQLFromFile(String filePath) {
    StringBuilder sqlBuilder = new StringBuilder();
    try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
      String line;
      while ((line = br.readLine()) != null) {
        sqlBuilder.append(line).append("\n");
      }
    } catch (IOException e) {
      e.printStackTrace();
      return null;
    }
    return removeComments(sqlBuilder.toString());
  }

  private String removeComments(String sql) {
    // Remove single-line comments
    String singleLineCommentRegex = "--.*?(\r?\n|$)";
    sql = sql.replaceAll(singleLineCommentRegex, "");

    // Remove multi-line comments
    String multiLineCommentRegex = "/\\*.*?\\*/";
    Pattern pattern = Pattern.compile(multiLineCommentRegex, Pattern.DOTALL);
    Matcher matcher = pattern.matcher(sql);
    sql = matcher.replaceAll("");

    return sql;
  }

  private void processSQLStatements(String sql) {
    try {
      String[] sqlStatements = sql.split(";");
      for (String sqlStatement : sqlStatements) {
        sqlStatement = sqlStatement.trim();
        if (!sqlStatement.isEmpty()) {
          Statement statement = CCJSqlParserUtil.parse(sqlStatement);
          if (statement instanceof CreateTable) {
            handleCreateTable((CreateTable) statement);
          } else if (statement instanceof Drop) {
            // Ignore DROP TABLE statements
            System.out.println("Ignoring DROP TABLE statement: " + statement);
          }
        }
      }
    } catch (Exception e) {
      // End of input or parsing error
      e.printStackTrace();
    }
  }

  private void handleCreateTable(CreateTable createTable) {
    // Get the list of column definitions
    List<ColumnDefinition> columnDefinitions = createTable.getColumnDefinitions();
    TableInfo currentTable = new TableInfo(createTable.getTable().getName());

    // Process each column definition
    for (ColumnDefinition columnDefinition : columnDefinitions) {
      ColDataType colDataType = columnDefinition.getColDataType();
      String dataType = colDataType.getDataType().toUpperCase();

      // Ignore the length/precision for VARCHAR, DECIMAL, etc.
      if (dataType.equals("VARCHAR") || dataType.equals("DECIMAL")) {
        colDataType.setArgumentsStringList(null);
      }

      currentTable.addColumn(
          new ColumnInfo(
              columnDefinition.getColumnName(),
              colDataType.toString(),
              isPrimaryKey(columnDefinition.getColumnSpecs())));
      System.out.println(
          "Column: " + columnDefinition.getColumnName() + ", Data Type: " + colDataType);
    }

    // Print the modified CREATE TABLE statement
    tables.add(currentTable);
    //    System.out.println("Modified CREATE TABLE SQL: " + createTable.toString());
  }

  private boolean isPrimaryKey(List<String> specs) {
    if (specs == null || specs.isEmpty()) {
      return false;
    }
    return String.join(" ", specs).toUpperCase().contains("PRIMARY KEY");
  }

  public ColumnType getColumnTypeByName(String tableName, String colName) {
    for (TableInfo table : tables) {
      if (table.getTableName().equalsIgnoreCase(tableName)) {
        return table.findColumnType(colName);
      }
    }
    return null;
  }

  public List<ColumnInfo> getColumnInfoByTableName(String tableName) {
    List<ColumnInfo> res = new ArrayList<>();
    for (TableInfo table : tables) {
      if (table.getTableName().equalsIgnoreCase(tableName)) {
        for (Map.Entry<String, ColumnInfo> entry : table.getColumns().entrySet()) {
          res.add(entry.getValue());
        }
        return res;
      }
    }
    return null;
  }

  @Override
  public String toString() {
    return "Schema{tables=" + tables + '}';
  }
}
