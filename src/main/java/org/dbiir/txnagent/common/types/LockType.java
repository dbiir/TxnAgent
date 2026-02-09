package org.dbiir.txnagent.common.types;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public enum LockType {
  NoneType("None"),
  EX("EX"),
  SH("SH");
  private final String name;

  LockType(String name) {
    this.name = name;
  }

  protected static final Map<Integer, LockType> idx_lookup = new HashMap<>();
  protected static final Map<String, LockType> name_lookup = new HashMap<>();

  static {
    for (LockType vt : EnumSet.allOf(LockType.class)) {
      LockType.idx_lookup.put(vt.ordinal(), vt);
      LockType.name_lookup.put(vt.name().toUpperCase(), vt);
    }
  }

  public String getName() {
    return name;
  }

  public static LockType get(String name) {
    return (LockType.name_lookup.get(name.toUpperCase()));
  }
}
