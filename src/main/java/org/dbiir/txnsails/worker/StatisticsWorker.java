package org.dbiir.txnsails.worker;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StatisticsWorker {
  public static class TransactionSample {
    private final long transactionId;
    private final Set<Integer> partitions;
    private final long startTime;
    private long endTime;
    @Getter private boolean committed;
    private final AtomicBoolean completed = new AtomicBoolean(false);

    public TransactionSample(long transactionId, Set<Integer> partitions) {
      this.transactionId = transactionId;
      this.partitions = new HashSet<>(partitions);
      this.startTime = System.currentTimeMillis();
    }

    public void addPartition(int partitionId) {
      this.partitions.add(partitionId);
    }

    public TransactionSample(long transactionId) {
      this.transactionId = transactionId;
      this.partitions = new HashSet<>(4);
      this.startTime = System.currentTimeMillis();
    }

    public void complete(boolean committed) {
      this.committed = committed;
      this.endTime = System.currentTimeMillis();
    }

    public long getLatency() {
      return endTime - startTime;
    }

    public Set<Integer> getPartitions() {
      return Collections.unmodifiableSet(partitions);
    }
  }

  private static class TimeWindowStats {
    private final long windowStart;
    private final AtomicInteger totalTransactions = new AtomicInteger(0);
    private final AtomicInteger committedTransactions = new AtomicInteger(0);
    private final AtomicLong totalLatency = new AtomicLong(0);
    private final ConcurrentMap<Integer, PartitionStats> partitionStats = new ConcurrentHashMap<>();

    public TimeWindowStats(long windowStart) {
      this.windowStart = windowStart;
    }

    public void addTransaction(TransactionSample transaction) {
      totalTransactions.incrementAndGet();
      if (transaction.isCommitted()) {
        committedTransactions.incrementAndGet();
      }
      totalLatency.addAndGet(transaction.getLatency());

      for (Integer partition : transaction.getPartitions()) {
        partitionStats.computeIfAbsent(partition, p -> new PartitionStats()).update(transaction);
      }
    }

    public int getThroughput() {
      return totalTransactions.get();
    }

    public double getAverageLatency() {
      int count = totalTransactions.get();
      if (count == 0) return 0.0;
      return (double) totalLatency.get() / count;
    }

    public double getRollbackRate() {
      int total = totalTransactions.get();
      if (total == 0) return 0.0;
      return 1.0 - ((double) committedTransactions.get() / total);
    }

    public double getPartitionRollbackRate(Integer partitionId) {
      PartitionStats stats = partitionStats.get(partitionId);
      return stats != null ? stats.getRollbackRate() : 0.0;
    }
  }

  private static class PartitionStats {
    private final AtomicInteger total = new AtomicInteger(0);
    private final AtomicInteger rollbacks = new AtomicInteger(0);

    public void update(TransactionSample transaction) {
      total.incrementAndGet();
      if (!transaction.isCommitted()) {
        rollbacks.incrementAndGet();
      }
    }

    public double getRollbackRate() {
      int t = total.get();
      if (t == 0) return 0.0;
      return (double) rollbacks.get() / t;
    }
  }

  private static Logger logger = LoggerFactory.getLogger(StatisticsWorker.class);
  private final DateTimeFormatter formatter =
      DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
  private final ConcurrentMap<Long, TimeWindowStats> timeWindowStats = new ConcurrentHashMap<>();
  private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
  private final AtomicLong currentWindow = new AtomicLong();
  private final BlockingQueue<TimeWindowStats> completedWindows = new LinkedBlockingQueue<>();
  private final int windowSizeMs = 1000; // 1 second
  private volatile boolean running = true;
  private final String filename;

  public StatisticsWorker(String directory) {
    initialize();
    // set unique filename by date and time
    this.filename = directory + "/statistics_" + LocalDateTime.now().format(formatter);
  }

  private void initialize() {
    long now = System.currentTimeMillis();
    currentWindow.set(now / windowSizeMs);

    scheduler.scheduleAtFixedRate(
        () -> {
          long currentTime = System.currentTimeMillis();
          long windowKey = currentTime / windowSizeMs;

          if (windowKey > currentWindow.get()) {
            TimeWindowStats oldStats = timeWindowStats.remove(currentWindow.get());
            if (oldStats != null) {
              completedWindows.offer(oldStats);
            }
            currentWindow.set(windowKey);
          }
        },
        100,
        100,
        TimeUnit.MILLISECONDS);

    scheduler.execute(
        () -> {
          while (running) {
            try {
              TimeWindowStats stats = completedWindows.poll(100, TimeUnit.MILLISECONDS);
              if (stats != null) {
                processCompletedWindow(stats);
              }
            } catch (InterruptedException e) {
              Thread.currentThread().interrupt();
              break;
            }
          }
        });
  }

  public void recordTransaction(TransactionSample transaction) {
    if (!transaction.completed.get()) {
      throw new IllegalStateException("Transaction not completed");
    }

    long windowKey = transaction.endTime / windowSizeMs;
    TimeWindowStats stats =
        timeWindowStats.computeIfAbsent(windowKey, k -> new TimeWindowStats(k * windowSizeMs));
    stats.addTransaction(transaction);
  }

  public int getCurrentThroughput() {
    TimeWindowStats stats = timeWindowStats.get(currentWindow.get());
    return stats != null ? stats.getThroughput() : 0;
  }

  public double getCurrentAverageLatency() {
    TimeWindowStats stats = timeWindowStats.get(currentWindow.get());
    return stats != null ? stats.getAverageLatency() : 0.0;
  }

  public double getCurrentRollbackRate() {
    TimeWindowStats stats = timeWindowStats.get(currentWindow.get());
    return stats != null ? stats.getRollbackRate() : 0.0;
  }

  public double getPartitionRollbackRate(int partitionId) {
    TimeWindowStats stats = timeWindowStats.get(currentWindow.get());
    return stats != null ? stats.getPartitionRollbackRate(partitionId) : 0.0;
  }

  /** return past N windows' throughput trend */
  public List<Integer> getThroughputTrend(int windowCount) {
    List<Integer> trend = new ArrayList<>();
    long current = currentWindow.get();

    for (int i = 0; i < windowCount; i++) {
      TimeWindowStats stats = timeWindowStats.get(current - i);
      trend.add(stats != null ? stats.getThroughput() : 0);
    }

    Collections.reverse(trend);
    return trend;
  }

  /** process completed window */
  private void processCompletedWindow(TimeWindowStats stats) {
    // print to this.filename in statistic directory in append mode
    String content =
        String.format(
            "Window [%s] processed: Throughput=%d, AvgLatency=%.2fms, RollbackRate=%.2f%%%n",
            Instant.ofEpochMilli(stats.windowStart),
            stats.getThroughput(),
            stats.getAverageLatency(),
            stats.getRollbackRate() * 100);

    try (FileWriter fw = new FileWriter(this.filename, true);
        BufferedWriter bw = new BufferedWriter(fw);
        PrintWriter out = new PrintWriter(bw)) {

      out.print(content);

    } catch (IOException e) {
      logger.error("Error while writing to file", e);
    }
  }

  /** close */
  public void shutdown() {
    running = false;
    scheduler.shutdown();
    try {
      if (!scheduler.awaitTermination(5, TimeUnit.SECONDS)) {
        scheduler.shutdownNow();
      }
    } catch (InterruptedException e) {
      scheduler.shutdownNow();
      Thread.currentThread().interrupt();
    }

    // process remaining windows
    for (TimeWindowStats stats : timeWindowStats.values()) {
      processCompletedWindow(stats);
    }
  }
}
