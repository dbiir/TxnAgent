package org.dbiir.txnagent.execution.validation;

import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.execution.sample.TransactionMeta;
import org.dbiir.txnagent.execution.utils.RWRecord;

public class TransactionCollector {
  private static final TransactionCollector INSTANCE;
  public static int TRANSACTION_BATCH = 250;
  private static final double SAMPLE_PROBABILITY = 0.01;
  private final Random random = new Random();
  static String edgeFormat = "#%d,%d,%d";
  private final Lock lock;
  @Getter @Setter private boolean needFlush = false;
  private TransactionMeta[] transactionMetas;
  private int sampleCount;

  static {
    INSTANCE = new TransactionCollector();
  }

  public TransactionCollector() {
    this.lock = new ReentrantLock();
    this.transactionMetas = new TransactionMeta[TRANSACTION_BATCH + 1];
    this.sampleCount = 0;
  }

  public void addTransactionSample(
      int transactionType, List<RWRecord> reads, List<RWRecord> writes, int processing) {
    lock.lock();
    if (!needFlush) {
      transactionMetas[sampleCount++] =
          new TransactionMeta(transactionType, reads, writes, processing);
      if (sampleCount >= TRANSACTION_BATCH) needFlush = true;
    }
    // System.out.println("sample count: " + sampleCount);
    lock.unlock();
  }

  public void refreshMetas() {
    lock.lock();
    sampleCount = 0;
    needFlush = false;
    lock.unlock();
  }

  public String getTransactionNodeFeature(int idx) {
    return transactionMetas[idx].transactionFeature();
  }

  public String getTransactionEdgeFeature(int idx) {
    StringBuilder builder = new StringBuilder();
    int table_idx;
    for (int i = 0; i < TRANSACTION_BATCH; i++) {
      if (i == idx) continue;
      if ((table_idx = rrDependency(idx, i)) > 0) {
        builder.append(edgeFormat.formatted(i, 1, table_idx));
      }
      if ((table_idx = rwDependency(idx, i)) > 0) {
        builder.append(edgeFormat.formatted(i, 2, table_idx));
      }
      if ((table_idx = wwDependency(idx, i)) > 0) {
        builder.append(edgeFormat.formatted(i, 4, table_idx));
      }
    }
    return builder.toString();
  }

  private int rrDependency(int idx1, int idx2) {
    for (RWRecord r : transactionMetas[idx2].getRset()) {
      for (RWRecord r2 : transactionMetas[idx1].getRset()) {
        if (r.table_idx() == r2.table_idx() && r.key_id() == r2.key_id()) return r.table_idx();
      }
    }
    return -1;
  }

  private int rwDependency(int idx1, int idx2) {
    for (RWRecord w : transactionMetas[idx2].getWset()) {
      for (RWRecord r : transactionMetas[idx1].getRset()) {
        if (w.table_idx() == r.table_idx() && w.key_id() == r.key_id()) return w.table_idx();
      }
    }
    for (RWRecord w : transactionMetas[idx1].getWset()) {
      for (RWRecord r : transactionMetas[idx2].getRset()) {
        if (w.table_idx() == r.table_idx() && w.key_id() == r.key_id()) return w.table_idx();
      }
    }
    return -1;
  }

  private int wwDependency(int idx1, int idx2) {
    for (RWRecord w1 : transactionMetas[idx2].getWset()) {
      for (RWRecord w2 : transactionMetas[idx1].getWset()) {
        if (w1.table_idx() == w2.table_idx() && w1.key_id() == w2.key_id()) return w1.table_idx();
      }
    }
    return -1;
  }

  public boolean isSample() {
    return !needFlush && random.nextDouble() < SAMPLE_PROBABILITY;
  }

  public static TransactionCollector getInstance() {
    return INSTANCE;
  }
}
