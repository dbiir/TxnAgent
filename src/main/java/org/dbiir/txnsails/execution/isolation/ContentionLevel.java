package org.dbiir.txnsails.execution.isolation;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public enum ContentionLevel {
  LOW("LOW"),
  MEDIUM("MEDIUM"),
  HIGH("HIGH");

  private final String level;
  ContentionLevel(String level) {
    this.level = level;
  }
  static final Map<Integer, ContentionLevel> idx_lookup = new HashMap<>();
  static final Map<String, ContentionLevel> name_lookup = new HashMap<>();

  static {
    for (ContentionLevel vt : EnumSet.allOf(ContentionLevel.class)) {
      ContentionLevel.idx_lookup.put(vt.ordinal(), vt);
      ContentionLevel.name_lookup.put(vt.name().toUpperCase(), vt);
    }
  }

  public String getLevel() {
    return level;
  }

  public static ContentionLevel get(String name) {
    return (ContentionLevel.name_lookup.get(name.toUpperCase()));
  }
}
