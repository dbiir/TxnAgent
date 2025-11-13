package org.dbiir.txnsails.execution.validation;

import java.sql.SQLException;
import java.util.HashMap;
import java.util.List;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnsails.analysis.ConditionInfo;
import org.dbiir.txnsails.common.TemplateSQL;
import org.dbiir.txnsails.common.constants.SmallBankConstants;
import org.dbiir.txnsails.common.constants.TPCCConstants;
import org.dbiir.txnsails.common.constants.YCSBConstants;
import org.dbiir.txnsails.common.types.CCType;
import org.dbiir.txnsails.common.types.LockType;
import org.dbiir.txnsails.execution.isolation.DataItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Getter
@Setter
public class ValidationMeta {
  private static final Logger logger = LoggerFactory.getLogger(ValidationMeta.class);
  private static final int SEED = 100000;
  private TemplateSQL templateSQL;
  /*
   * Change the type of oldVersions to List<> for scan operation in the future,
   * we do not support this because the absence of interval validation lock in the middle tier,
   * which would be implemented in the next iteration version and has little impact on our contribution.
   * TODO: We would refer to some existing mature technique solutions, such as SIREAD Lock in PostgreSQL.
   */
  private long oldVersion;
  // params for placeholder
  private int uniqueKeyNumber;
  private int idForValidation; // identification for validation (mod INT_MAX for hash value)
  // key -> int
  private HashMap<String, Integer> uniqueKeys = new HashMap<>(4);
  private boolean locked = false;
  private LockType lockType;
  private String relationName;
  private DataItem dataItem;

  public void setTemplateSQL(TemplateSQL templateSQL) {
    this.templateSQL = templateSQL;
    this.lockType = templateSQL.getOp() == 0 ? LockType.SH : LockType.EX;
    this.relationName = templateSQL.getTable();
  }

  /*
   * @return true if there are extra args for local 2PC optimization
   */
  public boolean addRuntimeArgs(List<String> args, int offset) {
    List<ConditionInfo> whereConditionInfos = templateSQL.getWherePlaceholders();
    if (args.size() - offset < whereConditionInfos.size()) {
      idForValidation = -1;
      return false;
    }
    for (ConditionInfo conditionInfo : whereConditionInfos) {
      int v = Integer.parseInt(args.get(offset + conditionInfo.getPlaceholderIndex()));
      uniqueKeys.put(conditionInfo.getUpperCaseColumnName(), v);
    }
    // calculate the id for validation
    switch (ValidationMetaTable.getInstance().getWorkload()) {
      case "smallbank":
        idForValidation = SmallBankConstants.calculateUniqueId(uniqueKeys, templateSQL.getTable());
        break;
      case "tpcc":
        idForValidation = TPCCConstants.calculateUniqueId(uniqueKeys, templateSQL.getTable());
        break;
      case "ycsb":
        idForValidation = YCSBConstants.calculateUniqueId(uniqueKeys, templateSQL.getTable());
        break;
      case "unknown benchmark":
        break;
    }

    // Optimization for local 2PC
    return args.size() - offset > whereConditionInfos.size();
  }

  public void clearInfo() {
    oldVersion = -1;
    uniqueKeyNumber = 0;
    idForValidation = -1;
    locked = false;
  }

  public void deepCopy(ValidationMeta other) {
    this.templateSQL = other.getTemplateSQL();
    this.idForValidation = other.getIdForValidation();
  }

  public void validateFS(long tid) throws SQLException {
    // add validation lock
    // System.out.println(Thread.currentThread().getName() + " tid #" + tid  + " try validation lock #" + key + " relation #" + relationName + " type: " + lockType);
    ValidationMetaTable.getInstance().tryValidationLock(relationName, tid, idForValidation, lockType, CCType.NUM_CC);
    if (lockType == LockType.EX) {
      return;
    }
    // validation
    long lastestVersion = ValidationMetaTable.getInstance().getHotspotVersion(relationName, idForValidation);
    if (lastestVersion < 0) {
      lastestVersion = ValidationMetaTable.getInstance().fetchUnknownVersionCache(relationName, idForValidation);
    }
    if (lastestVersion != oldVersion) {
      String msg = String.format("Validation failed for key #%d, %s, lastestVersion: %d, oldVersion: %d", idForValidation, relationName, lastestVersion, oldVersion);
      logger.error(msg);
      throw new SQLException(msg, "500");
    }
  }

  public void doAfterCommit(long tid, boolean isCommit) {
    if (isCommit && lockType == LockType.EX) {
      ValidationMetaTable.getInstance().updateHotspotVersion(relationName, idForValidation, oldVersion);
      // System.out.println(Thread.currentThread().getName() + " update #" + key + " relation #" + relationName + " oldVersion: " + oldVersion);
    }
    ValidationMetaTable.getInstance().releaseValidationLock(relationName, idForValidation, lockType);
    // System.out.println(Thread.currentThread().getName() + " release #" + key + " relation #" + relationName + " type: " + lockType);
  }

  public void releaseValidationLock(long tid) {
    ValidationMetaTable.getInstance().releaseValidationLock(relationName, idForValidation, lockType);
  }

  public void copy(ValidationMeta item) {
    this.relationName = item.relationName;
    this.idForValidation = item.idForValidation;
    this.oldVersion = item.oldVersion;
    this.lockType = item.lockType;
  }

  public void copy(String relationName, int idForValidation, LockType lockType, long oldVersion) {
    this.relationName = relationName;
    this.idForValidation = idForValidation;
    this.oldVersion = oldVersion;
    this.lockType = lockType;
  }
}
