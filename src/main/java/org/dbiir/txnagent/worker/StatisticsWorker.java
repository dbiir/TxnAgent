package org.dbiir.txnagent.worker;

import java.io.*;
import java.net.Socket;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.SneakyThrows;
import org.dbiir.txnagent.common.constants.SmallBankConstants;
import org.dbiir.txnagent.common.constants.TPCCConstants;
import org.dbiir.txnagent.common.constants.YCSBConstants;
import org.dbiir.txnagent.common.types.IsolationLevelType;
import org.dbiir.txnagent.execution.isolation.Participant;
import org.dbiir.txnagent.execution.isolation.PartitionManager;
import org.dbiir.txnagent.execution.isolation.Transaction;
import org.dbiir.txnagent.execution.validation.ValidationSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StatisticsWorker implements Runnable {
  private static final Logger logger = LoggerFactory.getLogger(StatisticsWorker.class);
  private static final double sampleProbability = 0.1;
  private static final String ip = "localhost";
  private static final int port = 7654;
  private static final int INTERVAL_SECONDS = 5;

  private RandomGenerator random;
  // cross transaction statistic
  private int maxPartitionId;
  private int[][] partitionRelations; // 2D list: distributed txn count between partitions
  private final List<PartitionStat> partitionStats = new ArrayList<>();
  private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();

  // ── Counters ──────────────────────────────────────────────────
  // committedCount: incremented for EVERY committed txn (100%).
  // Used by the 1s scheduler to compute per-second throughput.
  // Reset by the scheduler each tick.
  private final AtomicInteger committedCount = new AtomicInteger(0);

  // abortedCount: incremented for EVERY aborted txn (100%).
  // Used to compute the interval-level abort ratio.
  // Reset each cycle by reset().
  private final AtomicInteger abortedCount = new AtomicInteger(0);

  // sampledTxnCount: incremented for 10%-sampled txns (committed OR aborted).
  // Used as the denominator for per-partition workloadIntensity.
  // Reset each cycle by reset().
  private final AtomicInteger sampledTxnCount = new AtomicInteger(0);
  // ──────────────────────────────────────────────────────────────

  private final List<Integer> throughputHistory = new ArrayList<>();
  private final AtomicLong lastReportTime = new AtomicLong(System.currentTimeMillis());
  private final String workload;

  // statistic file sent to transactions
  private final String filepath;
  @Setter
  private String outputFilePrefix;
  private Socket socket;
  private PrintWriter out;
  private BufferedReader in;
  private static String headFormat = "%d#%d#%.2f#%.2f";
  private static String nodeFormat = "%d#%.4f#%.4f#%.4f#%d";
  private static String edgeFormat = "%d#%d#%d";
  private static String infoRequestFormat = "online,%s";

  public StatisticsWorker(String filepath, String workload) {
    this.filepath = filepath;
    this.workload = workload;
    startThroughputRecording();
    this.random = RandomGenerator.getDefault();
    initPartitionRelations();
    initPartitionStats();
    try {
      this.socket = new Socket(ip, port);
      this.out = new PrintWriter(socket.getOutputStream(), true);
      this.in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
    } catch (Exception e) {
      logger.error("Failed to connect to flusher socket at {}:{}", ip, port);
    }
  }

  @SneakyThrows
  @Override
  public void run() {
    // Wait for the first interval so we have meaningful data before the first
    // collection
    try {
      Thread.sleep(INTERVAL_SECONDS * 1000L);
    } catch (InterruptedException ignored) {
      return;
    }

    while (!Thread.currentThread().isInterrupted()) {
      long timestamp = System.currentTimeMillis();
      String fileName = filepath + "/sample_" + timestamp;
      logger.info(fileName);
      try (FileWriter fileWriter = new FileWriter(fileName, true)) {
        fileWriter.write(getPartitionStats());
        fileWriter.close();
        logger.info("Content appended to the file.");
      } catch (IOException ex) {
        logger.error("An error occurred: " + ex.getMessage());
        throw ex;
      }

      // send to agent
      out.println(infoRequestFormat.formatted(fileName));
      reset();

      // wait for reply and apply actions
      logger.debug("Send the request to the server: online,predict,{}", fileName);
      String data = in.readLine();
      logger.debug("Receive the prediction result: {}", data);

      applyActions(data);

      // sleep for next round
      try {
        Thread.sleep(INTERVAL_SECONDS * 1000L);
      } catch (InterruptedException ignored) {
        return;
      }
    }
    // Notify Python agent to shut down
    try {
      out.println("close");
      socket.close();
    } catch (IOException e) {
      logger.error("Failed to send close command: {}", e.getMessage());
    }
    stopThroughputRecording();
  }

  private void getMaxPartitionId() {
    switch (workload) {
      case "ycsb" -> {
        maxPartitionId = YCSBConstants.getGlobalPartitionIds().size();
      }
      case "smallbank" -> {
        maxPartitionId = SmallBankConstants.getGlobalPartitionIds().size();
      }
      case "tpcc" -> {
        maxPartitionId = TPCCConstants.getGlobalPartitionIds().size();
      }
      default -> {
      }
    }
  }

  private void initPartitionRelations() {
    getMaxPartitionId();
    partitionRelations = new int[maxPartitionId][maxPartitionId];
  }

  private void initPartitionStats() {
    getMaxPartitionId();
    for (int i = 0; i < maxPartitionId; i++) {
      partitionStats.add(new PartitionStat(i));
    }
  }

  /**
   * Called by transaction execution threads for every completed transaction.
   *
   * @param transaction the transaction that just finished
   * @param committed   true if committed, false if aborted
   */
  public void recordTransaction(Transaction transaction, boolean committed) {
    // 100% counters — used for throughput and abort ratio
    if (committed) {
      committedCount.incrementAndGet();
    } else {
      abortedCount.incrementAndGet();
    }

    // sampling — used for per-partition read/write/abort stats
    boolean sample = random.nextDouble() < sampleProbability;
    if (!sample) {
      return;
    }
    sampledTxnCount.incrementAndGet();

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
    for (Integer partitionId : partitions) {
      for (Integer otherPartitionId : partitions) {
        if (partitionId.equals(otherPartitionId)) {
          partitionRelations[partitionId][otherPartitionId] += 1;
          partitionRelations[otherPartitionId][partitionId] += 1;
        }
      }
      if (committed) {
        partitionStats.get(partitionId).getCommitCount().incrementAndGet();
      } else {
        partitionStats.get(partitionId).getAbortCount().incrementAndGet();
      }
    }
  }

  private String getPartitionStats() {
    // head: nodeCount#edgeCount#throughput#abortRate
    // node: id#readCount#writeCount#abortRatio#workloadIntensive#isolationLevel
    // edge: id1#id2#count
    StringBuilder nodes = new StringBuilder();
    for (int i = 0; i < maxPartitionId; i++) {
      PartitionStat stat = partitionStats.get(i);
      nodes
          .append(
              String.format(
                  nodeFormat,
                  i,
                  stat.getReadRatio(),
                  stat.getAbortRatio(),
                  stat.getWorkloadIntensity(sampledTxnCount.get()),
                  transferIsolationLevelToInt(PartitionManager.getInstance().getIsolation(i))))
          .append("\n");
    }

    StringBuilder edges = new StringBuilder();
    int edgeCount = 0;
    for (int i = 0; i < maxPartitionId; i++) {
      for (int j = i + 1; j < maxPartitionId; j++) {
        if (partitionRelations[i][j] > 0) {
          edges.append(String.format(edgeFormat, i, j, partitionRelations[i][j])).append("\n");
          edgeCount++;
        }
      }
    }
    // Average throughput over last 5 seconds (matches the adjustment interval)
    double avgThroughput = 0;
    int windowSize = Math.min(throughputHistory.size(), INTERVAL_SECONDS);
    if (windowSize > 0) {
      for (int k = throughputHistory.size() - windowSize; k < throughputHistory.size(); k++) {
        avgThroughput += throughputHistory.get(k);
      }
      avgThroughput /= windowSize;
    }
    // Per-interval abort ratio: aborted / (committed + aborted)
    int totalThisInterval = committedCount.get() + abortedCount.get();
    double abortRatio = totalThisInterval > 0 ? 1.0 * abortedCount.get() / totalThisInterval : 0.0;

    String head = String.format(headFormat, maxPartitionId, edgeCount, avgThroughput, abortRatio);
    return head + "\n" + nodes + edges;
  }

  private int transferIsolationLevelToInt(IsolationLevelType level) {
    return switch (level) {
      case SER -> 2;
      case SI -> 1;
      case RC -> 0;
      default -> -1;
    };
  }

  private void startThroughputRecording() {
    scheduler.scheduleAtFixedRate(
        () -> {
          long now = System.currentTimeMillis();
          long startTime = lastReportTime.getAndSet(now);
          long elapsed = now - startTime;

          // Compute per-second throughput and reset the counter
          int committed = committedCount.getAndSet(0);
          double throughput = 1000.0 * committed / Math.max(elapsed, 1);
          throughputHistory.add((int) throughput);
          if (throughputHistory.size() > 60) {
            throughputHistory.removeFirst();
          }
        },
        10,
        1,
        TimeUnit.SECONDS);
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

  private void reset() {
    partitionRelations = new int[maxPartitionId][maxPartitionId];
    for (PartitionStat stat : partitionStats) {
      stat.reset();
    }
    sampledTxnCount.set(0);
    abortedCount.set(0);
  }

  /*
   * Response format from Python agent: id#iso#mu;id#iso#mu;...
   * iso: 0=RC, 1=SI, 2=SER
   * mu: integer
   */
  private void applyActions(String data) {
    String[] parts = data.split(";");
    for (String part : parts) {
      String[] items = part.split("#");
      int partitionId = Integer.parseInt(items[0]);
      int isoInt = Integer.parseInt(items[1]);
      int mu = Integer.parseInt(items[2]);

      IsolationLevelType level = switch (isoInt) {
        case 0 -> IsolationLevelType.RC;
        case 1 -> IsolationLevelType.SI;
        case 2 -> IsolationLevelType.SER;
        default -> {
          logger.error("Unknown isolation level: {}", isoInt);
          yield null;
        }
      };
      if (level != null) {
        PartitionManager.getInstance().setIsolation(partitionId, level);
      }
      PartitionManager.getInstance().setMu(partitionId, mu);
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
      if (total == 0)
        return 0.0f;
      return (float) readCount.get() / total;
    }

    public float getWorkloadIntensity(int totalSampledTxns) {
      if (totalSampledTxns == 0)
        return 0.0f;
      return 1.0f * (abortCount.get() + commitCount.get()) / totalSampledTxns;
    }

    public float getAbortRatio() {
      return 1.0f * abortCount.get() / Math.max(abortCount.get() + commitCount.get(), 1);
    }

    public void reset() {
      readCount.set(0);
      writeCount.set(0);
      abortCount.set(0);
      commitCount.set(0);
    }
  }
}
