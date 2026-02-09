package org.dbiir.txnagent.common.types;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

/**
 * List of the database management systems that we support in the framework.
 *
 * @author pavlo
 */
public enum DatabaseType {
  H2(true, false),
  HSQLDB(false, false),
  POSTGRES(false, false, true),
  MYSQL(true, false);

  DatabaseType(
      boolean escapeNames, boolean includeColNames, boolean loadNeedsUpdateColumnSequence) {
    this.escapeNames = escapeNames;
    this.includeColNames = includeColNames;
    this.loadNeedsUpdateColumnSequence = loadNeedsUpdateColumnSequence;
  }

  DatabaseType(boolean escapeNames, boolean includeColNames) {
    this(escapeNames, includeColNames, false);
  }

  /** If this flag is set to true, then the framework will escape names in the INSERT queries */
  private final boolean escapeNames;

  /**
   * If this flag is set to true, then the framework will include the column names when generating
   * INSERT queries for loading data.
   */
  private final boolean includeColNames;

  /**
   * If this flag is set to true, the framework will attempt to update the column sequence after
   * loading data.
   */
  private final boolean loadNeedsUpdateColumnSequence;

  // ---------------------------------------------------------------
  // ACCESSORS
  // ----------------------------------------------------------------

  /**
   * @return True if the framework should escape the names of columns/tables when generating SQL to
   *     load in data for the target database type.
   */
  public boolean shouldEscapeNames() {
    return (this.escapeNames);
  }

  /**
   * @return True if the framework should include the names of columns when generating SQL to load
   *     in data for the target database type.
   */
  public boolean shouldIncludeColumnNames() {
    return (this.includeColNames);
  }

  /**
   * @return True if the framework should attempt to update the column sequence after loading data.
   */
  public boolean shouldUpdateColumnSequenceAfterLoad() {
    return (this.loadNeedsUpdateColumnSequence);
  }

  // ----------------------------------------------------------------
  // STATIC METHODS + MEMBERS
  // ----------------------------------------------------------------

  protected static final Map<Integer, DatabaseType> idx_lookup = new HashMap<>();
  protected static final Map<String, DatabaseType> name_lookup = new HashMap<>();

  static {
    for (DatabaseType vt : EnumSet.allOf(DatabaseType.class)) {
      DatabaseType.idx_lookup.put(vt.ordinal(), vt);
      DatabaseType.name_lookup.put(vt.name().toUpperCase(), vt);
    }
  }

  public static DatabaseType get(String name) {
    return (DatabaseType.name_lookup.get(name.toUpperCase()));
  }
}
