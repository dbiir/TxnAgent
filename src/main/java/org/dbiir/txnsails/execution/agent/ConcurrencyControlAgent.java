package org.dbiir.txnsails.execution.agent;

import lombok.Getter;
import org.dbiir.txnsails.execution.isolation.Transaction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConcurrencyControlAgent {
  private static final Logger logger= LoggerFactory.getLogger(ConcurrencyControlAgent.class);
  private static final ConcurrencyControlAgent INSTANCE;
  private long mu;

  static {
    INSTANCE = new ConcurrencyControlAgent();
  }

  public ConcurrencyControlAgent() {
    // TODO:
  }

  public long getMu(Transaction t_i, Transaction t_j) {
    // TODO:
    return 0;
  }

  public long getCommitTimestamp(Transaction transaction) {
    return transaction.getLowerBound();
  }

  public static ConcurrencyControlAgent getInstance() {
    return INSTANCE;
  }
}
