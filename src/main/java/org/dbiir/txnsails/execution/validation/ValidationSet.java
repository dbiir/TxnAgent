package org.dbiir.txnsails.execution.validation;

import java.util.ArrayList;
import java.util.List;
import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ValidationSet {
  private static final Logger logger = LoggerFactory.getLogger(ValidationSet.class);
  @Getter private final List<ValidationMeta> items;
  @Getter private int itemCount;

  public ValidationSet(int capacity) {
    this.items = new ArrayList<>(capacity);
    this.itemCount = 0;
  }

  public void addValidationMetas(List<ValidationMeta> validationItems) {
    this.items.addAll(validationItems);
    this.itemCount += validationItems.size();
  }

  public void addValidationMeta(ValidationMeta validationItem) {
    this.items.add(validationItem);
    this.itemCount++;
  }

  public ValidationMeta get(int idx) {
    if (idx >= this.itemCount) {
      logger.error("index out of bound");
      return null;
    }
    return this.items.get(idx);
  }

  public void reset() {
    this.itemCount = 0;
    this.items.clear();
  }

  public boolean isEmpty() {
    return itemCount == 0;
  }
}
