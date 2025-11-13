package org.dbiir.txnsails.common.types;

import lombok.Getter;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

@Getter
public enum IsolationLevelType {
  SER("SERIALIZABLE"),
  RC("RC"),
  SI("SI"),
  NUM_CC("NUM_CC");
  private final String name;

  IsolationLevelType(String name) {
    this.name = name;
  }

  static final Map<Integer, IsolationLevelType> idx_lookup = new HashMap<>();
  static final Map<String, IsolationLevelType> name_lookup = new HashMap<>();

  static {
    for (IsolationLevelType vt : EnumSet.allOf(IsolationLevelType.class)) {
      IsolationLevelType.idx_lookup.put(vt.ordinal(), vt);
      IsolationLevelType.name_lookup.put(vt.name().toUpperCase(), vt);
    }
  }

  public static IsolationLevelType get(String name) {
    return (IsolationLevelType.name_lookup.get(name.toUpperCase()));
  }
}
