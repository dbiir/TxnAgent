package org.dbiir.txnsails.execution.validation;

import java.util.HashMap;
import java.util.List;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnsails.analysis.ConditionInfo;
import org.dbiir.txnsails.common.TemplateSQL;
import org.dbiir.txnsails.common.constants.SmallBankConstants;
import org.dbiir.txnsails.common.constants.TPCCConstants;
import org.dbiir.txnsails.common.constants.YCSBConstants;

@Getter
@Setter
public class ValidationMeta {
  private static final int SEED = 100000;
  private TemplateSQL templateSQL;
  /*
   * Change the type of oldVersions to List<> for scan operation in the future,
   * we do not support this because the absence of interval validation lock in the middle tier,
   * which would be implemented in the next iteration version and has little impact on our contribution.
   * TODO: We would refer to some existing mature technique solutions, such as SIREAD Lock in PostgreSQL.
   */
  private int oldVersions;
  // params for placeholder
  private int uniqueKeyNumber;
  private int idForValidation; // identification for validation (mod INT_MAX for hash value)
  // key -> int
  private HashMap<String, Integer> uniqueKeys = new HashMap<>(4);
  private boolean locked = false;

  public void addRuntimeArgs(List<String> args, int offset) {
    List<ConditionInfo> whereConditionInfos = templateSQL.getWherePlaceholders();
    if (args.size() - offset < whereConditionInfos.size()) {
      idForValidation = -1;
      return;
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
  }

  public void clearInfo() {
    oldVersions = -1;
    uniqueKeyNumber = 0;
    idForValidation = -1;
    locked = false;
  }

  public void deepCopy(ValidationMeta other) {
    this.templateSQL = other.getTemplateSQL();
    this.idForValidation = other.getIdForValidation();
  }
}
