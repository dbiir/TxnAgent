package org.dbiir.txnsails.worker;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import java.nio.charset.StandardCharsets;
import java.sql.*;
import java.text.MessageFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import lombok.Getter;
import lombok.Setter;
import net.sf.jsqlparser.schema.Column;
import org.apache.commons.lang3.StringUtils;
import org.dbiir.txnsails.analysis.ConditionInfo;
import org.dbiir.txnsails.analysis.SchemaInfo;
import org.dbiir.txnsails.common.*;
import org.dbiir.txnsails.common.types.CCType;
import org.dbiir.txnsails.common.types.ColumnType;
import org.dbiir.txnsails.common.types.LockType;
import org.dbiir.txnsails.execution.WorkloadConfiguration;
import org.dbiir.txnsails.execution.isolation.*;
import org.dbiir.txnsails.execution.utils.RWRecord;
import org.dbiir.txnsails.execution.utils.SQLStmt;
import org.dbiir.txnsails.execution.utils.TransactionIdGenerator;
import org.dbiir.txnsails.execution.validation.TransactionCollector;
import org.dbiir.txnsails.execution.validation.ValidationMeta;
import org.dbiir.txnsails.execution.validation.ValidationMetaTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class OnlineWorker implements Runnable {
  private static final Logger logger = LoggerFactory.getLogger(OnlineWorker.class);
  private static final int MAX_ISOLATION_LEVEL = 3;
  protected Connection conn = null;
  private WorkloadConfiguration configuration = null;
  private final Random random = new Random();
  @Getter private final int id;
  CCType ccType;
  @Setter @Getter protected boolean switchFinish = false;

  @Getter @Setter
  protected boolean switchPhaseReady = false; // validation transaction common, set it to true

  @Getter @Setter private TransactionStatus status = TransactionStatus.IDLE;
  private HashMap<String, TransactionTemplate> templates = new HashMap<>();
  // support at most 20 validation entries in a single transaction
  private static final int MAX_VALIDATION_META = 20;
  private static final double SAMPLE_PROBABILITY = 0.01;
  private static final long mask = 0x7FFFFFFFFFFFFFFFL;
  private final ValidationMeta[] validationMetaUnderRC = new ValidationMeta[MAX_VALIDATION_META];
  private final ValidationMeta[] validationMetaUnderSI = new ValidationMeta[MAX_VALIDATION_META];
  private int validationMetaIdxUnderRC = 0;
  private int validationMetaIdxUnderSI = 0;
  private boolean shouldSample = false;
  private int transactionId;
  // variables for sampling
  private final List<RWRecord> readSet;
  private final List<RWRecord> writeSet;
  private ValidationMeta sampleMeta = new ValidationMeta();
  private CCType lockManner = CCType.SER;
  private final SchemaInfo schema;
  // variables for netty
  private final ChannelHandlerContext ctx;
  // variables for fine serializable scheduling
  private Connection[] connections = new Connection[3];
  private boolean[] connectionUsed = new boolean[3];
  private Participant[] participants = new Participant[3];
  private AsyncResultWrapper[] resultList = new AsyncResultWrapper[3];
  private final ValidationMeta[] validationMetaFS = new ValidationMeta[MAX_VALIDATION_META];
  private int validationMetaFSIdx = 0;
  private final Transaction transaction;
  private boolean lastestSQL;

  public OnlineWorker(WorkloadConfiguration configuration, int id, ChannelHandlerContext context) {
    this.configuration = configuration;
    this.id = id;
    this.ctx = context;
    this.ccType = configuration.getConcurrencyControlType();

    // init the connection
    try {
      if (ccType == CCType.FS) {
        Connection conn = makeConnection();
        conn.setAutoCommit(true);
        conn.setTransactionIsolation(Connection.TRANSACTION_READ_COMMITTED);
        this.connections[0] = conn; // 0: READ COMMITTED
        Connection conn2 = makeConnection();
        conn2.setAutoCommit(true);
        conn2.setTransactionIsolation(Connection.TRANSACTION_REPEATABLE_READ);
        this.connections[1] = conn2; // 1: REPEATABLE READ
        Connection conn3 = makeConnection();
        conn3.setAutoCommit(true);
        conn3.setTransactionIsolation(Connection.TRANSACTION_SERIALIZABLE);
        this.connections[2] = conn3; // 2: SERIALIZABLE
      } else {
        this.conn = makeConnection();
        this.conn.setAutoCommit(false);
        switch (ccType) {
          case RC, RC_TAILOR -> this.conn.setTransactionIsolation(
              Connection.TRANSACTION_READ_COMMITTED);
          case SI, SI_TAILOR -> this.conn.setTransactionIsolation(
              Connection.TRANSACTION_REPEATABLE_READ);
          case SER -> this.conn.setTransactionIsolation(Connection.TRANSACTION_SERIALIZABLE);
        }
      }
    } catch (SQLException ex) {
      throw new RuntimeException("Failed to connect to database", ex);
    }
    // clone the analysis result to thread local
    MetaWorker.getINSTANCE().cloneTemplatesAfterAnalysis(this.templates);
    // init the validation metas
    for (int i = 0; i < MAX_VALIDATION_META; i++) {
      if (ccType == CCType.FS) {
        this.validationMetaFS[i] = new ValidationMeta();
      } else {
        this.validationMetaUnderRC[i] = new ValidationMeta();
        this.validationMetaUnderSI[i] = new ValidationMeta();
      }
    }
    // init the first transactionID
    this.transactionId =
        (int) (((System.nanoTime() << 10) | (Thread.currentThread().threadId() & 0x3ff)) & mask);
    // init sample container
    this.readSet = new ArrayList<>(8);
    this.writeSet = new ArrayList<>(8);
    // fetch the schema
    this.schema = MetaWorker.getINSTANCE().getSchema();
    // register self into AdapterWorker
    Adapter.getInstance().addOnlineWorker(this);
    System.out.println(this.toString() + " is initialized.");
    // init the transaction for fine serializable scheduling
    this.transaction = new Transaction(this.transactionId);
    for (int i = 0; i < MAX_ISOLATION_LEVEL; i++) {
      this.participants[i] = new Participant(i, connections[i]);
    }
  }

  private Connection makeConnection() throws SQLException {
    if (StringUtils.isEmpty(configuration.getUsername())) {
      return DriverManager.getConnection(configuration.getUrl());
    } else {
      return DriverManager.getConnection(
          configuration.getUrl(), configuration.getUsername(), configuration.getPassword());
    }
  }

  public void beginFS() {
    long tid = TransactionIdGenerator.generateTransactionId(id);
    this.transaction.init(tid);
    TransactionManager.getInstance().addTransaction(this.transaction);
    logger.info("{} transaction #{} begin.", Thread.currentThread().getName(), tid);
  }

  /**
   * online execution
   *
   * @param args - args[0]: transaction template name - args[1]: sql index in this template -
   *     args[2:]: the params for execution - args[-1]: if exists, presents the lastest query
   * @return execution results
   */
  public String executeFS(String[] args, int offset) throws SQLException {
    if (args.length < offset) return "";
    StringBuilder results = new StringBuilder();
    TemplateSQL templateSQL =
        this.templates
            .get(args[offset - 2])
            .getSQLTemplateByIndex(Integer.parseInt(args[offset - 1]));

    // record the sql that need validate
    ValidationMeta meta = this.validationMetaFS[validationMetaFSIdx++];
    meta.setTemplateSQL(templateSQL);
    if (templateSQL.getUniqueKeyNumber() <= args.length - offset) {
      lastestSQL = meta.addRuntimeArgs(List.of(args), offset);
      System.out.println(
          this.toString()
              + templateSQL.getSQL()
              + ", "
              + Arrays.asList(args).subList(offset, args.length)
              + ", Id for validation: "
              + meta.getIdForValidation());
      // TODO: sampling
      if (shouldSample) {
        addSampleMeta(
            templateSQL.getOp(),
            MetaWorker.getINSTANCE().getRelationType(templateSQL.getTable()),
            meta.getIdForValidation());
      }
    } else {
      System.out.println("The number of real time args is not enough.");
      throw new SQLException("Not enough arguments!");
    }

    // find the isolation level
    int isolationLevel = PartitionManager.getInstance().chooseIsolation(meta);

    String executeSQL = templateSQL.getSQL();
    boolean firstUse = false;
    if (!connectionUsed[isolationLevel]) {
      // append `BEGIN' statement to this connection
      executeSQL = "BEGIN; " + executeSQL;
      connectionUsed[isolationLevel] = true;
      this.transaction.addParticipant(this.participants[isolationLevel]);
      firstUse = true;
    }

    // execute the sql
    try (PreparedStatement stmtc =
        this.getPreparedStatement(conn, new SQLStmt(executeSQL), args, offset, templateSQL)) {
      stmtc.setQueryTimeout(1);

      try (ResultSet rs = firstUse ? omitBeginStatement(stmtc) : stmtc.executeQuery()) {
        int v = -1;
        List<List<String>> rows = new ArrayList<>(2);
        while (rs.next()) {
          List<String> row = new ArrayList<>();
          for (Column col : templateSQL.getColumnList()) {
            String columnName = col.getColumnName();
            if (columnName.equalsIgnoreCase("vid")) {
              v = rs.getInt(columnName);
            } else {
              row.add(rs.getString(columnName));
            }
          }
          rows.add(row);
        }
        // parse and wrap the results
        results.append(wrapResults(rows));

        // record the version if it needs, support scan-based
        validationMetaFS[validationMetaFSIdx - 1].setOldVersion(v);
        logger.info("{} {} execute finished", this.toString(), executeSQL);
      } catch (SQLException ex) {
        // check if the error can retry automatically, in the future
        logger.error("{} failed to execute sql: {}", this.toString(), executeSQL);
        throw ex;
      }
    }

    // validate

    if (templateSQL.getOp() == 0) {
      // read operation
      DataItem item = PartitionManager.getInstance().getAndAddDataItem(meta);
      item.read(transaction, meta.getOldVersion());
      meta.setDataItem(item);
      TransactionManager.getInstance().read(this.transaction, meta);
      this.participants[isolationLevel].addReadValidationMeta(meta);
    } else if (templateSQL.getOp() == 1) {
      // write operation
      TransactionManager.getInstance().write(this.transaction, meta);
    }

    return results.toString();
  }

  private ResultSet omitBeginStatement(PreparedStatement stmt) throws SQLException {
    boolean hasResultSet = stmt.execute();
    ResultSet finalResultSet = null;
    do {
      if (hasResultSet) {
        finalResultSet = stmt.getResultSet();
        break;
      }
      hasResultSet = stmt.getMoreResults();
    } while (hasResultSet || stmt.getUpdateCount() != -1);

    return finalResultSet;
  }

  /**
   * online execution
   *
   * @param args
   * @return execution results args[0]: transaction template name args[1]: sql index in this
   *     template args[2:]: the params for
   */
  public String execute(String[] args, int offset) throws SQLException {
    if (args.length < offset) return "";
    StringBuilder results = new StringBuilder();
    TemplateSQL templateSQL =
        this.templates
            .get(args[offset - 2])
            .getSQLTemplateByIndex(Integer.parseInt(args[offset - 1]));

    // record the sql that need validate
    if (templateSQL.isNeedRewriteUnderRC() || templateSQL.isNeedRewriteUnderSI() || shouldSample) {
      ValidationMeta meta =
          templateSQL.isNeedRewriteUnderRC()
              ? validationMetaUnderRC[validationMetaIdxUnderRC]
              : templateSQL.isNeedRewriteUnderSI()
                  ? validationMetaUnderSI[validationMetaIdxUnderSI]
                  : sampleMeta;
      meta.setTemplateSQL(templateSQL);
      if (templateSQL.getUniqueKeyNumber() <= args.length - offset) {
        meta.addRuntimeArgs(List.of(args), offset);
        System.out.println(
            this.toString()
                + templateSQL.getSQL()
                + ", "
                + Arrays.asList(args).subList(offset, args.length)
                + ", Id for validation: "
                + meta.getIdForValidation());

        if (shouldSample) {
          addSampleMeta(
              templateSQL.getOp(),
              MetaWorker.getINSTANCE().getRelationType(templateSQL.getTable()),
              meta.getIdForValidation());
        }
        if (templateSQL.isNeedRewriteUnderRC()) {
          validationMetaIdxUnderRC++;
          if (templateSQL.isNeedRewriteUnderSI()) {
            validationMetaUnderSI[validationMetaIdxUnderSI++].deepCopy(meta);
          }
        }
      } else {
        System.out.println("The number of real time args is not enough.");
        throw new SQLException("Not enough arguments!");
      }
    }

    String executeSQL = templateSQL.getSQL();
    // execute the sql
    try (PreparedStatement stmtc =
        this.getPreparedStatement(conn, new SQLStmt(executeSQL), args, offset, templateSQL)) {
      stmtc.setQueryTimeout(1);
      try (ResultSet rs = stmtc.executeQuery()) {
        int v = -1;
        List<List<String>> rows = new ArrayList<>(2);
        while (rs.next()) {
          List<String> row = new ArrayList<>();
          for (Column col : templateSQL.getColumnList()) {
            String columnName = col.getColumnName();
            if (columnName.equalsIgnoreCase("vid")) {
              v = rs.getInt(columnName);
            } else {
              row.add(rs.getString(columnName));
            }
          }
          rows.add(row);
        }
        // parse and wrap the results
        results.append(wrapResults(rows));
        // record the version if it needs, support scan-based
        if (templateSQL.isNeedRewriteUnderRC()) {
          validationMetaUnderRC[validationMetaIdxUnderRC - 1].setOldVersion(v);
          if (templateSQL.isNeedRewriteUnderSI()) {
            validationMetaUnderSI[validationMetaIdxUnderSI - 1].setOldVersion(v);
          }
        }
        System.out.println(this.toString() + " execute finished");
      } catch (SQLException ex) {
        // check if the error can retry automatically, in the future
        System.out.println(this.toString() + "Error execute sql: " + executeSQL);
        System.out.println(ex);
        throw ex;
      }
    }

    return results.toString();
  }

  public void commit() throws SQLException {
    /* validate before commitment, release validation locks after commitment */
    try {
      System.out.println(this.toString() + " is validating");
      validate();
      System.out.println(this.toString() + " has validated, is committing");
      conn.commit();
      System.out.println(this.toString() + " has committed");
      releaseValidationLocks(true);
      if (shouldSample) sampleTransaction(true);
    } catch (SQLException ex) {
      releaseValidationLocks(false);
      rollback();
      if (shouldSample) sampleTransaction(false);
      throw ex;
    }

    clearPreviousTransactionInfo();
    /* sample transaction if txnSails needs and choose whether sample next transaction */
    shouldSample = random.nextDouble() < SAMPLE_PROBABILITY;
    this.transactionId =
        (int) (((System.nanoTime() << 10) | (Thread.currentThread().threadId() & 0x3ff)) & mask);
    // switch the isolation mode
    if (Adapter.getInstance().isInSwitchPhase()) {
      System.out.println(this.toString() + " enters switch phase");
      switchConnectionIsolationMode();
    }
    System.out.println(this.toString() + " return commit()");
  }

  public void commitFS() throws SQLException {
    // 1. check the async prepare results
    if (!this.transaction.isPrepared()) {
      rollbackFS();
      return;
    }
    // 2. commit or rollback a transaction
    TransactionManager.getInstance().commit(this.transaction, this.resultList);
    for (AsyncResultWrapper result : this.resultList) {
      if (!result.isSuccess()) {
        logger.error(
            "{} transaction #{} commit failed after preparation, {}",
            Thread.currentThread().getName(),
            this.transaction.getId(),
            result.getException().getMessage());
      }
    }

    // 3. reset the transaction meta
    resetTransactionMeta();

    // 4. remove transaction from TransactionManager
    TransactionManager.getInstance().removeTransaction(this.transaction);
  }

  public void rollback() throws SQLException {
    try {
      System.out.println(this.toString() + " is rollbacking");
      conn.rollback();
      System.out.println(this.toString() + " has been rollbacked");
    } finally {
      clearPreviousTransactionInfo();
      /* sample transaction if txnSails needs and choose whether sample next transaction */
      shouldSample = random.nextDouble() < SAMPLE_PROBABILITY;
      this.transactionId =
          (int) (((System.nanoTime() << 10) | (Thread.currentThread().threadId() & 0x3ff)) & mask);
      // switch the isolation
      if (Adapter.getInstance().isInSwitchPhase()) {
        switchConnectionIsolationMode();
      }
    }
  }

  public void rollbackFS() throws SQLException {
    assert !this.transaction.isPrepared();
    TransactionManager.getInstance().rollback(this.transaction, this.resultList);
    for (AsyncResultWrapper result : this.resultList) {
      if (!result.isSuccess()) {
        logger.error(
            "{} transaction #{} rollback failed, {}",
            Thread.currentThread().getName(),
            this.transaction.getId(),
            result.getException().getMessage());
        // reset the connection for this participant
      }
    }

    // reset the transaction meta
    resetTransactionMeta();

    // remove transaction from TransactionManager
    TransactionManager.getInstance().removeTransaction(this.transaction);
  }

  private void resetTransactionMeta() {
    this.validationMetaFSIdx = 0;
    for (int i = 0; i < MAX_ISOLATION_LEVEL; i++) {
      this.connectionUsed[i] = false;
      this.participants[i].reset();
      this.resultList[i].reset();
    }
    this.transaction.reset();
  }

  private void validate() throws SQLException {
    /* 0. block transaction during the transition until
     * 1. acquire validation lock (transition: acquire the lock according the stricter isolation level)
     * 2. check the version
     */
    while (Adapter.getInstance().isInSwitchPhase()
        && !Adapter.getInstance().isAllWorkersReadyForSwitch()) {
      // set current thread ready, block for all thread to ready
      if (!this.switchPhaseReady) {
        this.switchPhaseReady = true;
        System.out.println(this.toString() + " is ready for switch");
      } else {
        try {
          Thread.sleep(5);
          // break;
        } catch (InterruptedException e) {
        }
      }
    }
    if (Adapter.getInstance().isInSwitchPhase() && this.ccType == CCType.SER) {
      // cases: SER -> SI/RC, SI/RC -> SER
      this.ccType = CCType.SER_TRANSITION;
    }
    if (!Adapter.getInstance().isInSwitchPhase() && this.ccType == CCType.SER_TRANSITION) {
      // cases: SI/RC -> SER
      this.ccType = CCType.SER;
    }

    this.lockManner = Adapter.getInstance().getCCType();
    if (lockManner == CCType.RC_TAILOR) {
      for (int i = 0; i < validationMetaIdxUnderRC; i++) {
        ValidationMeta validationMeta = this.validationMetaUnderRC[i];
        TemplateSQL templateSQL = validationMeta.getTemplateSQL();
        LockType lockType = templateSQL.getOp() == 0 ? LockType.SH : LockType.EX;
        ValidationMetaTable.getInstance()
            .tryValidationLock(
                templateSQL.getTable(),
                this.transactionId,
                validationMeta.getIdForValidation(),
                lockType,
                Adapter.getInstance().getCCType());
        System.out.println(this.toString() + " validated " + validationMeta.getIdForValidation());
        validationMeta.setLocked(true);
      }
    } else if (lockManner == CCType.SI_TAILOR) {
      for (int i = 0; i < validationMetaIdxUnderSI; i++) {
        ValidationMeta validationMeta = this.validationMetaUnderSI[i];
        TemplateSQL templateSQL = validationMeta.getTemplateSQL();
        LockType lockType = templateSQL.getOp() == 0 ? LockType.SH : LockType.EX;
        ValidationMetaTable.getInstance()
            .tryValidationLock(
                templateSQL.getTable(),
                this.transactionId,
                validationMeta.getIdForValidation(),
                lockType,
                Adapter.getInstance().getCCType());
        System.out.println(this.toString() + " validated " + validationMeta.getIdForValidation());
        validationMeta.setLocked(true);
      }
    }

    // validation row versions
    if (ccType == CCType.SI_TAILOR || ccType == CCType.SER_TRANSITION) {
      for (int i = 0; i < validationMetaIdxUnderSI; i++) {
        validateSingleMeta(validationMetaUnderSI[i]);
      }
    } else if (ccType == CCType.RC_TAILOR) {
      for (int i = 0; i < validationMetaIdxUnderRC; i++) {
        validateSingleMeta(validationMetaUnderRC[i]);
      }
    }
  }

  private void validateSingleMeta(ValidationMeta meta) throws SQLException {
    long v =
        ValidationMetaTable.getInstance()
            .getHotspotVersion(meta.getTemplateSQL().getTable(), (long) meta.getIdForValidation());
    if (v >= 0) {
      if (v != meta.getOldVersion()) {
        String msg =
            String.format(
                "Validation failed for key #%d, %s",
                meta.getIdForValidation(), meta.getTemplateSQL().getTable());
        throw new SQLException(msg, "500");
      }
    } else {
      System.out.println(this.toString() + " fetch unknown version");
      v =
          ValidationMetaTable.getInstance()
              .fetchUnknownVersionCache(
                  meta.getTemplateSQL().getTable(), meta.getIdForValidation());
      if (v != meta.getOldVersion()) {
        //          releaseValidationLocks(false);
        String msg =
            String.format(
                "Validation failed for ycsb_key #%d, usertable", meta.getIdForValidation());
        throw new SQLException(msg, "500");
      }
      System.out.println(this.toString() + " fetch unknown version down");
    }
  }

  private void clearPreviousTransactionInfo() {
    this.validationMetaIdxUnderRC = 0;
    this.validationMetaIdxUnderSI = 0;
  }

  private void addSampleMeta(int op, int relationType, int idx) {
    if (op == 0) {
      this.readSet.add(new RWRecord(relationType, idx));
    } else if (op == 1) {
      this.writeSet.add(new RWRecord(relationType, idx));
    }
  }

  private void releaseValidationLocks(boolean success) {
    if (this.lockManner == CCType.RC_TAILOR) {
      for (int i = 0; i < validationMetaIdxUnderRC; i++) {
        ValidationMeta meta = validationMetaUnderRC[i];
        if (!meta.isLocked()) {
          if (success) updateValidationVersion(meta);
          continue;
        }
        releaseSingleMeta(meta, success);
      }
    } else if (this.lockManner == CCType.SI_TAILOR) {
      for (int i = 0; i < validationMetaIdxUnderSI; i++) {
        ValidationMeta meta = validationMetaUnderSI[i];
        if (!meta.isLocked()) {
          if (success) updateValidationVersion(meta);
          continue;
        }
        releaseSingleMeta(meta, success);
      }
    }
  }

  private void updateValidationVersion(ValidationMeta meta) {
    TemplateSQL templateSQL = meta.getTemplateSQL();
    LockType lockType = templateSQL.getOp() == 0 ? LockType.SH : LockType.EX;
    if (lockType == LockType.EX) {
      // update the hot version cache (HVC) if the entry is cached in memory
      ValidationMetaTable.getInstance()
          .updateHotspotVersion(
              templateSQL.getTable(), meta.getIdForValidation(), meta.getOldVersion() + 1);
    }
    meta.clearInfo();
  }

  private void releaseSingleMeta(ValidationMeta meta, boolean success) {
    if (!meta.isLocked()) {
      System.out.println("Lock operation is failed, but previous validation successes");
    }
    TemplateSQL templateSQL = meta.getTemplateSQL();
    LockType lockType = templateSQL.getOp() == 0 ? LockType.SH : LockType.EX;
    ValidationMetaTable.getInstance()
        .releaseValidationLock(templateSQL.getTable(), meta.getIdForValidation(), lockType);
    if (success && lockType == LockType.EX) {
      // update the hot version cache (HVC) if the entry is cached in memory
      ValidationMetaTable.getInstance()
          .updateHotspotVersion(
              templateSQL.getTable(), meta.getIdForValidation(), meta.getOldVersion() + 1);
    }
    meta.setLocked(false);
    meta.clearInfo();
  }

  private boolean shouldRewrite(TemplateSQL templateSQL, CCType t) {
    if ((t == CCType.SI_TAILOR || t == CCType.SER_TRANSITION)
        && templateSQL.isNeedRewriteUnderSI()) {
      return true;
    }

    if (t == CCType.RC_TAILOR && templateSQL.isNeedRewriteUnderRC()) {
      return true;
    }
    return false;
  }

  // mode: Connection.TRANSACTION_READ_COMMITTED, TRANSACTION_REPEATABLE_READ,
  // TRANSACTION_SERIALIZABLE
  private void switchConnectionIsolationMode() throws SQLException {
    // block for all validate transaction completed
    while (Adapter.getInstance().isInSwitchPhase()
        && !Adapter.getInstance().isAllWorkersReadyForSwitch()) {
      if (!this.switchPhaseReady) {
        this.switchPhaseReady = true;
        System.out.println(this.toString() + " is ready for switch");
      } else {
        try {
          Thread.sleep(5);
          // break;
        } catch (InterruptedException ignored) {
        }
      }
    }

    if (Adapter.getInstance().isInSwitchPhase() && !switchFinish) {
      this.ccType = Adapter.getInstance().getNextCCType();

      switch (this.ccType) {
        case RC, RC_TAILOR -> conn.setTransactionIsolation(Connection.TRANSACTION_READ_COMMITTED);
        case SI, SI_TAILOR -> conn.setTransactionIsolation(Connection.TRANSACTION_REPEATABLE_READ);
        case SER, SER_TRANSITION -> conn.setTransactionIsolation(
            Connection.TRANSACTION_SERIALIZABLE);
      }
      if (this.ccType == CCType.SER) {
        this.ccType = CCType.SER_TRANSITION;
      }
      switchFinish = true;
      conn.setAutoCommit(false);
      System.out.println(
          Thread.currentThread().getName()
              + "switch finish: "
              + Adapter.getInstance().getNextCCType());
    }
  }

  private void sampleTransaction(boolean success) {
    TransactionCollector.getInstance().addTransactionSample(1, readSet, writeSet, success ? 1 : 0);
  }

  /**
   * Return a PreparedStatement for the given SQLStmt handle The underlying Procedure API will make
   * sure that the proper SQL for the target DBMS is used for this SQLStmt. This will automatically
   * call setObject for all the parameters you pass in
   *
   * @param conn
   * @param stmt
   * @param params: just support String, maybe provide more types in the future
   * @param templateSQL
   * @return
   * @throws SQLException
   */
  private final PreparedStatement getPreparedStatement(
      Connection conn, SQLStmt stmt, String[] params, int offset, TemplateSQL templateSQL)
      throws SQLException {
    PreparedStatement pStmt = conn.prepareStatement(stmt.getSQL());
    if (params.length - offset != templateSQL.getAllPlaceholders().size()) {
      String msg = "The length of parameters and placeholders are not matching";
      System.out.println(
          msg
              + "; params.length: "
              + params.length
              + " templateSQL: "
              + templateSQL.getAllPlaceholders().size());
      throw new SQLException(msg);
    }
    List<ConditionInfo> phList = templateSQL.getAllPlaceholders();
    int idx = 1;
    for (int i = offset; i < params.length; i++) {
      // rewrite by the type of the params
      ColumnType columnType =
          schema.getColumnTypeByName(templateSQL.getTable(), phList.get(idx - 1).getColumnName());
      switch (columnType) {
        case INTEGER -> pStmt.setObject(idx, Integer.valueOf(params[i]));
        case BIGINT -> pStmt.setObject(idx, Long.valueOf(params[i]));
        case FLOAT -> pStmt.setObject(idx, Float.valueOf(params[i]));
        case VARCHAR, TEXT -> pStmt.setObject(idx, params[i]);
        case DOUBLE -> pStmt.setObject(idx, Double.valueOf(params[i]));
        case BOOLEAN -> pStmt.setObject(idx, Boolean.valueOf(params[i]));
        default -> throw new AssertionError();
      }
      idx++;
    }
    return (pStmt);
  }

  private String wrapResults(List<List<String>> rows) {
    StringBuilder sb = new StringBuilder();

    for (List<String> row : rows) {
      sb.append(String.format("%02x", row.size())); // record the column size in this row
      for (String col : row) {
        // record the string length in 4 bytes
        //        System.out.println("col.length() = " + col.length());
        sb.append(String.format("%04x", col.length()));
        sb.append(col); // wrap column value
      }
    }
    return sb.toString();
  }

  private List<List<String>> dewrapResults(String results) {
    List<List<String>> rows = new ArrayList<>();
    int index = 0;

    while (index < results.length()) {
      // decode the number of columns in this row
      int count = Integer.parseInt(results.substring(index, index + 2), 16);
      index += 2;

      List<String> row = new ArrayList<>();
      for (int i = 0; i < count; i++) {
        // read 4 char as the column length (2 bytes)
        String lengthHex = results.substring(index, index + 4);
        int length = Integer.parseInt(lengthHex, 16);
        index += 4;

        String value = results.substring(index, index + length);
        index += length; // move to the next

        row.add(value);
      }
      rows.add(row); // add row
    }

    return rows;
  }

  @Override
  public String toString() {
    return String.format("%s<%03d>", this.getClass().getSimpleName(), this.getId());
  }

  @Override
  public void run() {
    while (!Thread.interrupted()) {
      String clientRequest = MetaWorker.getINSTANCE().getExecutionMessage(id);
      if (clientRequest != null) {
        // execute the request
        String response = "";
        String[] parts = clientRequest.split("#");
        for (int i = 0; i < parts.length; i++) {
          parts[i] = parts[i].trim();
        }
        String functionName = parts[0].toLowerCase(); // function name is case-insensitive
        try {
          switch (functionName) {
            case "execute" -> {
              response = ccType == CCType.FS ? executeFS(parts, 3) : execute(parts, 3);
              response = "OK#" + response;
            }
            case "begin" -> {
              if (ccType == CCType.FS) {
                beginFS();
              } else {
                // no op for normal cc
              }
              response = "OK";
            }
            case "commit" -> {
              if (ccType == CCType.FS) {
                commitFS();
              } else {
                commit();
              }
              response = "OK";
            }
            case "rollback" -> {
              if (ccType == CCType.FS) {
                rollbackFS();
              } else {
                rollback();
              }
              response = "OK";
            }
            default -> {
              System.out.println(functionName + " can not be handled by OnlineWorker");
            }
          }
        } catch (SQLException ex) {
          response =
              MessageFormat.format(
                  MetaWorker.ERROR_FORMATTER, ex.getMessage(), ex.getSQLState(), ex.getErrorCode());
        }

        // reset the client signal
        MetaWorker.getINSTANCE().resetMessageSignal(id);
        // send the response
        try {
          sendMessage(response);
        } catch (InterruptedException e) {
          System.out.println(Arrays.stream(e.getStackTrace()));
          ctx.close();
          return;
        }

        try {
          if (lastestSQL) {
            // prepare
            TransactionManager.getInstance().prepare(this.transaction);
            this.transaction.setPrepared(true);
          }
        } catch (SQLException ex) {
          this.transaction.setPrepared(false);
          logger.error(
              "{} failed to prepare transaction: {}",
              Thread.currentThread().getName(),
              ex.getMessage());
        }
      }
    }
    Adapter.getInstance().removeOnlineWorker(id);

    // close the connections when exit
    if (ccType == CCType.FS) {
      for (Connection c : connections) {
        try {
          c.close();
        } catch (SQLException e) {
          throw new RuntimeException(e);
        }
      }
    } else {
      try {
        conn.close();
      } catch (SQLException e) {
        throw new RuntimeException(e);
      }
    }
    System.out.println(this.toString() + " is out!");
  }

  private void sendMessage(String msg) throws InterruptedException {
    System.out.println(this.toString() + " sending: " + msg);
    ByteBuf resp = ctx.alloc().buffer(msg.length());
    resp.writeBytes(msg.getBytes(StandardCharsets.UTF_8));
    ctx.writeAndFlush(resp).sync();
  }
}
