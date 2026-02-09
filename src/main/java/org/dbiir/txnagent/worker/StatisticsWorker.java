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
  private RandomGenerator random;
  // cross transaction statistic
  private int maxPartitionId;
  private int[][] partitionRelations; // 2D list, with its elements represents distributed relations
  private final List<PartitionStat> partitionStats = new ArrayList<>();
  private final ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
  private final AtomicInteger totalTransactions = new AtomicInteger(0);
  private final AtomicInteger abortTransactions = new AtomicInteger(0);
  private final List<Integer> throughputHistory = new ArrayList<>();
  private final AtomicInteger currentSecondTransactionCount = new AtomicInteger(0);
  private final AtomicInteger currentSecondAbort = new AtomicInteger(0);
  private final AtomicLong lastReportTime = new AtomicLong(System.currentTimeMillis());
  private final String workload;

  // statistic file sent to transactions
  private final String filepath;
  @Setter private String outputFilePrefix;
  private Socket socket;
  private boolean online;
  private static String headFormat = "%d#%d#%.2f#%.2f";
  private static String nodeFormat = "%d#%d#%d#%.2f#%.2f#%d";
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
      this.online = true;
    } catch (Exception e) {
      this.online = false;
      logger.error("Failed to connect to flusher socket at {}:{}", ip, port);
    }
  }

  @SneakyThrows
  @Override
  public void run() {
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

      // collect system metrics and send to agent
      PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
      BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
      // format: global throughput, global abortRate, filename
      out.println(infoRequestFormat.formatted(fileName));
      // reset counters
      reset();

      // wait for reply and apply actions
      logger.debug("Send the request to the server: online,predict,{}", fileName);
      String data = in.readLine();
      logger.debug("Receive the prediction result: {}", data);

      // set partition isolation levels based on the response from agent
      applyActions(data);

      // sleep for next round
      try {
        Thread.sleep(1000);
      } catch (InterruptedException ignored) {
        return;
      }
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
      default -> {}
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

  public void recordTransaction(Transaction transaction, boolean committed) {
    // record the transaction for throughput calculation
    if (committed) {
      currentSecondTransactionCount.incrementAndGet();
    } else {
      abortTransactions.incrementAndGet();
    }

    boolean sample = random.nextDouble() < sampleProbability;
    if (!sample) {
      return;
    }
    totalTransactions.incrementAndGet();

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
                  stat.getReadCount().get(),
                  stat.getWriteCount().get(),
                  stat.getAbortRatio(),
                  stat.getWorkloadIntensity(totalTransactions.get()),
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
    String head =
        String.format(
            headFormat,
            maxPartitionId,
            edgeCount,
            throughputHistory.getLast(),
            1.0 * abortTransactions.get() / Math.max(totalTransactions.get(), 1));
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

          double throughput = 1.0 * currentSecondTransactionCount.get() / Math.max(elapsed, 1);
          throughputHistory.add((int) throughput);
          if (throughputHistory.size() > 60) {
            throughputHistory.removeFirst();
          }
        },
        10,
        1,
        TimeUnit.SECONDS); // at fixed rate of 1 second
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
  }

  /*
   * action type:
   *  - set isolation level [2]: 0-RC, 1-SI, 2-SER
   *  - set mu [3]: integer
   */
  private void applyActions(String data) {
    // partitionId, actionType, parameters
    String[] parts = data.split(";");
    for (String part : parts) {
      String[] items = part.split("#");
      int partitionId = Integer.parseInt(items[0]);
      int actionType = Integer.parseInt(items[1]);
      switch (actionType) {
        case 2 -> {
          int levelInt = Integer.parseInt(items[2]);
          IsolationLevelType level = null;
          switch (levelInt) {
            case 0 -> level = IsolationLevelType.RC;
            case 1 -> level = IsolationLevelType.SI;
            case 2 -> level = IsolationLevelType.SER;
            default -> logger.error("Unknown isolation level: {}", levelInt);
          }
          PartitionManager.getInstance().setIsolation(partitionId, level);
        }
        case 3 -> {
          int mu = Integer.parseInt(items[2]);
          PartitionManager.getInstance().setMu(partitionId, mu);
        }
        default -> logger.error("Unknown action type: {}", actionType);
      }
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
