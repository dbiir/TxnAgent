package org.dbiir.txnsails.execution.validation;

import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import org.dbiir.txnsails.common.types.LockType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ValidationSet {
  private static Logger logger = LoggerFactory.getLogger(ValidationSet.class);
  private final List<ValidationMeta> items;
  @Getter private int itemCount;

  public ValidationSet(int capacity) {
    this.items = new ArrayList<>(capacity);
    this.itemCount = 0;
    for (int i = 0; i < capacity; i++) {
      this.items.add(new ValidationMeta());
    }
  }

  public void addValidationMetas(List<ValidationMeta> validationItems) {
    for (ValidationMeta item : validationItems) {
      this.items.get(itemCount).copy(item);
      this.itemCount++;
    }
  }

  public void addValidationMeta(ValidationMeta validationItem) {
    this.items.get(itemCount).copy(validationItem);
    this.itemCount++;
  }

  public void addValidationMeta(String relationName, int key, LockType lockType, long oldVersion) {
    this.items.get(itemCount).copy(relationName, key, lockType, oldVersion);
    this.itemCount++;
  }

  public List<ValidationMeta> getValidationMetas() {
    return this.items.subList(0, itemCount);
  }

  public ValidationMeta get(int idx) {
    return this.items.get(idx);
  }

  public void reset() {
    this.itemCount = 0;
  }

  public boolean isEmpty() {
    return itemCount == 0;
  }
}
