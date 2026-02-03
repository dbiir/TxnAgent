package org.dbiir.txnsails.worker;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.dbiir.txnsails.common.constants.SmallBankConstants;
import org.dbiir.txnsails.common.constants.TPCCConstants;
import org.dbiir.txnsails.common.constants.YCSBConstants;
import org.dbiir.txnsails.execution.isolation.Participant;
import org.dbiir.txnsails.execution.isolation.Transaction;
import org.dbiir.txnsails.execution.validation.ValidationSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StatisticsWorker {
  private static final Logger logger = LoggerFactory.getLogger(StatisticsWorker.class);
  private static final double sampleProbability = 0.1;
  private RandomGenerator random;
  // cross transaction statistic
  private final HashMap<Integer, HashMap<Integer, Integer>> partitionRelations = new HashMap<>();
  private final HashMap<Integer, PartitionStat> partitionStats = new HashMap<>();
  private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
  private final AtomicInteger totalTransactions = new AtomicInteger(0);
  private final List<Integer> throughputHistory = new ArrayList<>();
  private final AtomicInteger currentSecondTransactionCount = new AtomicInteger(0);
  private final AtomicLong lastReportTime = new AtomicLong(System.currentTimeMillis());
  private final String filepath;
  private final String workload;

  public StatisticsWorker(String filepath, String workload) {
    this.filepath = filepath;
    this.workload = workload;
    startThroughputRecording();
    this.random = RandomGenerator.getDefault();
    initPartitionStats();
  }

  private void initPartitionStats() {
    switch (workload) {
      case "ycsb" -> {
        for (Integer partitionId : YCSBConstants.getGlobalPartitionIds()) {
          this.partitionStats.put(partitionId, new PartitionStat(partitionId));
        }
      }
      case "smallbank" -> {
        for (Integer partitionId : SmallBankConstants.getGlobalPartitionIds()) {
          this.partitionStats.put(partitionId, new PartitionStat(partitionId));
        }
      }
      case "tpcc" -> {
        for (Integer partitionId : TPCCConstants.getGlobalPartitionIds()) {
          this.partitionStats.put(partitionId, new PartitionStat(partitionId));
        }
      }
      default -> {}
    }
  }

  public void shutdown() {
    stopThroughputRecording();
  }

  public void recordTransaction(Transaction transaction, boolean committed) {
    // record the transaction for throughput calculation
    currentSecondTransactionCount.incrementAndGet();

    boolean sample = random.nextDouble() < sampleProbability;
    if (!sample) {
      return;
    }
    totalTransactions.incrementAndGet();

    //    HashMap<String, Set<Integer>> partitions = new HashMap<>();
    HashSet<Integer> partitions = new HashSet<>();

    for (Participant p : transaction.getParticipants()) {
      ValidationSet readSet = p.getReadSet();
      int readItemCount = readSet.getItemCount();
      for (int i = 0; i < readItemCount; i++) {
        int partitionId = readSet.get(i).getDataItem().getPartitionId();
        partitions.add(partitionId);
        partitionStats.get(partitionId).getReadCount().incrementAndGet();
      }

      ValidationSet writeSet = p.getWriteSet();
      int writeItemCount = writeSet.getItemCount();
      for (int i = 0; i < writeItemCount; i++) {
        int partitionId = writeSet.get(i).getDataItem().getPartitionId();
        partitions.add(partitionId);
        partitionStats.get(partitionId).getWriteCount().incrementAndGet();
      }
    }

    // record partition relations
    Map<Integer, List<Integer>> relation = new HashMap<>();
    Integer[] array = partitions.toArray(new Integer[0]);

    for (int i = 0; i < array.length; i++) {
      List<Integer> partners = new ArrayList<>();
      for (int j = 0; j < array.length; j++) {
        if (i != j) {
          partners.add(array[j]);
        }
      }
      relation.put(array[i], partners);
    }

    // TODO: add into partitionRelations
  }

  private void startThroughputRecording() {
    scheduler.scheduleAtFixedRate(
        () -> {
          long now = System.currentTimeMillis();
          long startTime = lastReportTime.getAndSet(now);
          long elapsed = now - startTime;

          double throughput = 1.0 * currentSecondTransactionCount.get() / Math.max(elapsed, 1);
          throughputHistory.add((int) throughput);
          if (throughputHistory.size() > 60) {
            throughputHistory.removeFirst();
          }
        },
        10,
        1,
        TimeUnit.SECONDS); // 每秒触发
  }

  private void stopThroughputRecording() {
    scheduler.shutdown();
    try {
      if (!scheduler.awaitTermination(1, TimeUnit.SECONDS)) {
        scheduler.shutdownNow();
      }
    } catch (InterruptedException e) {
      scheduler.shutdownNow();
    }
  }

  @NoArgsConstructor
  @Getter
  @Setter
  private static class PartitionStat {
    private int partitionId;
    private AtomicInteger readCount = new AtomicInteger(0);
    private AtomicInteger writeCount = new AtomicInteger(0);
    private AtomicInteger abortCount = new AtomicInteger(0);
    private AtomicInteger commitCount = new AtomicInteger(0);

    public PartitionStat(int partitionId) {
      this.partitionId = partitionId;
    }

    public float getReadRatio() {
      int total = readCount.get() + writeCount.get();
      if (total == 0) return 0.0f;
      return (float) readCount.get() / total;
    }

    public float getWorkloadIntensity(int totalTransactions) {
      return 1.0f * (abortCount.get() + commitCount.get()) / totalTransactions;
    }

    public int getAbortCount() {
      return abortCount.get();
    }

    public void reset() {
      readCount.set(0);
      writeCount.set(0);
      abortCount.set(0);
      commitCount.set(0);
    }
  }
}
