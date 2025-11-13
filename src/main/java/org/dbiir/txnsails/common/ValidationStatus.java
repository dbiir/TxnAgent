package org.dbiir.txnsails.common;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public enum ValidationStatus {
  VALIDATED("VALIDATED"),
  VALIDATING("VALIDATING"),
  FAILED("FAILED"),
  VALIDATION_STATUS_NUM("VALIDATION_STATUS_NUM");
  private final String name;

  ValidationStatus(String name) {
    this.name = name;
  }

  static final Map<Integer, ValidationStatus> idx_lookup = new HashMap<>();
  static final Map<String, ValidationStatus> name_lookup = new HashMap<>();

  static {
    for (ValidationStatus vt : EnumSet.allOf(ValidationStatus.class)) {
      ValidationStatus.idx_lookup.put(vt.ordinal(), vt);
      ValidationStatus.name_lookup.put(vt.name().toUpperCase(), vt);
    }
  }

  public String getName() {
    return name;
  }

  public static ValidationStatus get(String name) {
    return (ValidationStatus.name_lookup.get(name.toUpperCase()));
  }
}
