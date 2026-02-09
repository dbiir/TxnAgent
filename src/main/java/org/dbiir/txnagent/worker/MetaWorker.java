package org.dbiir.txnagent.worker;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.analysis.ChordAbsentCycleFinder;
import org.dbiir.txnagent.analysis.SchemaInfo;
import org.dbiir.txnagent.common.*;
import org.dbiir.txnagent.common.constants.SmallBankConstants;
import org.dbiir.txnagent.common.constants.TPCCConstants;
import org.dbiir.txnagent.common.constants.YCSBConstants;
import org.dbiir.txnagent.common.types.DependencyType;

public class MetaWorker {
  @Getter private static MetaWorker INSTANCE = new MetaWorker();
  private HashMap<String, TransactionTemplate> templates = new HashMap<>();
  private StaticDependencyGraph sdg = new StaticDependencyGraph();
  private AtomicInteger globalTransactionGenerator;
  private int globalTemplateTypeGenerator;
  private final HashMap<String, Integer> relationEncodeMap = new HashMap<>();
  @Getter @Setter private SchemaInfo schema;

  // create the shared netty context array
  private static final int MAX_CLIENT_CONNECTIONS = 200;
  private AtomicBoolean[] messageSignalPool = new AtomicBoolean[MAX_CLIENT_CONNECTIONS];
  private String[] messagePool = new String[MAX_CLIENT_CONNECTIONS];

  // core affinity
  public static int MAX_AVAILABLE_CORES = 1;

  // error handling
  public static final String ERROR_FORMATTER = "ERROR#{0}#{1}#{2}"; // reason, SQLState, vendorCode

  public MetaWorker() {
    globalTemplateTypeGenerator = 1;
    registerTable();
  }

  public synchronized void registerTemplateName(String template_name) {
    templates.computeIfAbsent(
        template_name, k -> new TransactionTemplate(template_name, globalTemplateTypeGenerator));
    globalTemplateTypeGenerator <<= 1;
  }

  public int registerTemplateSQL(String template_name, int op, String relation, String sql) {
    templates.computeIfAbsent(template_name, k -> new TransactionTemplate(template_name));
    return templates.get(template_name).addTemplateSQL(op, relation, sql);
  }

  public int registerTemplateSQL(
      String template_name, int op, String relation, String sql, int idx) {
    templates.computeIfAbsent(template_name, k -> new TransactionTemplate(template_name));
    templates.get(template_name).setSkipIndex(idx, true);
    return templates.get(template_name).addTemplateSQL(op, relation, sql);
  }

  public int getRelationType(String relationName) {
    return this.relationEncodeMap.get(relationName);
  }

  private void registerTable() {
    this.relationEncodeMap.putAll(YCSBConstants.TABLENAME_TO_INDEX);
    this.relationEncodeMap.putAll(SmallBankConstants.TABLENAME_TO_INDEX);
    this.relationEncodeMap.putAll(TPCCConstants.TABLENAME_TO_INDEX);
  }

  /**
   * analysis the workload: - find the anomalies under SI and RC along with corresponding sql and
   * templates
   */
  public void analysisWorkload() {
    // construct the static dependency graph
    this.sdg = new StaticDependencyGraph();
    for (Map.Entry<String, TransactionTemplate> entry : templates.entrySet()) {
      sdg.addTemplate(entry.getValue());
    }
    sdg.addAllDependenciesByTemplates();

    ChordAbsentCycleFinder finder = new ChordAbsentCycleFinder(sdg.getAdjacencyList());
    System.out.println("============ find cycles ============");
    Set<StaticDependencyCycle> allCycles = finder.findAllCycles();
    for (StaticDependencyCycle cycle : allCycles) {
      System.out.print("Cycle: ");
      System.out.println(cycle);
    }

    // analysis RC
    System.out.println("============ find cycles ============");
    Set<StaticDependencyCycle> cyclesWithReadWrite = findCyclesWithReadWrite(allCycles);
    // output the result
    System.out.println("====== RC ======");
    for (StaticDependencyCycle cycle : cyclesWithReadWrite) {
      System.out.println(cycle);
    }
    for (StaticDependencyCycle cycle : cyclesWithReadWrite) {
      // Section 4.1, mark the needRewriteFlagUnderRC in template involved in RW dependency
      for (StaticDependencyGraphEdge edge : cycle.edges()) {
        if (edge.getType() == DependencyType.READ_WRITE) {
          edge.getFrom().setSQLRewriteByIndex(edge.getIdxInFrom(), "RC");
          edge.getTo().setSQLRewriteByIndex(edge.getIdxInTo(), "RC");
        }
      }
    }
    // analysis SI
    System.out.println("====== SI ======");
    Set<StaticDependencyCycle> cyclesWithTwoConsecutiveReadWrite =
        findCyclesWithTwoConsecutiveReadWrite(allCycles);
    for (StaticDependencyCycle cycle : cyclesWithTwoConsecutiveReadWrite) {
      System.out.println(cycle);
    }
    for (StaticDependencyCycle cycle : cyclesWithTwoConsecutiveReadWrite) {
      // Section 4.1, mark the needRewriteFlagUnderRC in template involved in second RW dependency
      List<StaticDependencyGraphEdge> edges = cycle.edges();
      for (int i = 0; i < edges.size(); i++) {
        if (edges.get(i).getType() == DependencyType.READ_WRITE
            && edges.get((i + 1) % edges.size()).getType() == DependencyType.READ_WRITE) {
          edges
              .get((i + 1) % edges.size())
              .getFrom()
              .setSQLRewriteByIndex(edges.get((i + 1) % edges.size()).getIdxInFrom(), "SI");
          edges
              .get((i + 1) % edges.size())
              .getTo()
              .setSQLRewriteByIndex(edges.get((i + 1) % edges.size()).getIdxInTo(), "SI");
        }
      }
    }
    System.out.println("========== find cycles end ==========");
  }

  private Set<StaticDependencyCycle> findCyclesWithReadWrite(Set<StaticDependencyCycle> allCycles) {
    Set<StaticDependencyCycle> result = new HashSet<>();

    for (StaticDependencyCycle cycle : allCycles) {
      for (StaticDependencyGraphEdge edge : cycle.edges()) {
        if (edge.getType() == DependencyType.READ_WRITE) {
          result.add(cycle);
          break; // find a cycle with one RW
        }
      }
    }
    return result;
  }

  private Set<StaticDependencyCycle> findCyclesWithTwoConsecutiveReadWrite(
      Set<StaticDependencyCycle> allCycles) {
    Set<StaticDependencyCycle> result = new HashSet<>();

    for (StaticDependencyCycle cycle : allCycles) {
      List<StaticDependencyGraphEdge> edges = cycle.edges();
      for (int i = 0; i < edges.size(); i++) {
        if (edges.get(i).getType() == DependencyType.READ_WRITE
            && edges.get((i + 1) % edges.size()).getType() == DependencyType.READ_WRITE) {
          result.add(cycle);
          break; // find a cycle with two consecutive RW
        }
      }
    }

    return result;
  }

  public void cloneTemplatesAfterAnalysis(HashMap<String, TransactionTemplate> cloneTemplates) {
    for (Map.Entry<String, TransactionTemplate> entry : templates.entrySet()) {
      cloneTemplates.put(entry.getKey(), entry.getValue().clone());
    }
  }

  public void flushAnalysisResult() {
    // for failure recovery
  }

  public static void setThreadAffinity(int coreId) {
    if (!Platform.isLinux()) {
      System.out.println("Setting thread affinity is supported only for Linux");
      return;
    }
    int pid = (int) Thread.currentThread().threadId();
    int longSize = Native.getNativeSize(Long.TYPE);
    int numLongs = (MAX_AVAILABLE_CORES + 63) / 64;
    Pointer cpuset = new Memory((long) longSize * numLongs);
    cpuset.clear((long) longSize * numLongs);

    // set the core
    int assignedCore = coreId % MAX_AVAILABLE_CORES;
    int longIndex = assignedCore / 64;
    int bitIndex = assignedCore % 64;
    cpuset.setLong((long) longIndex * longSize, 1L << bitIndex);

    int result = CLibrary.INSTANCE.sched_setaffinity(pid, longSize * numLongs, cpuset);
    if (result != 0) {
      throw new RuntimeException("Failed to set thread affinity: " + Native.getLastError());
    }
  }

  public static void setThreadAffinityMacOS(int coreId) {
    System.out.println("Setting thread affinity is not supported on macOS.");
  }

  public void initExecutionContext(int idx) {
    if (idx >= MAX_CLIENT_CONNECTIONS) {
      String msg = "too many connectors";
      System.out.println(msg);
      throw new RuntimeException(msg);
    }
    messageSignalPool[idx] = new AtomicBoolean(false);
    messagePool[idx] = "";
  }

  public String getExecutionMessage(int idx) {
    if (!messageSignalPool[idx].get()) {
      return null;
    } else {
      return messagePool[idx];
    }
  }

  public void addExecutionMessage(int idx, String msg) {
    if (messageSignalPool[idx].get()) {
      String errMsg = "can not execute before previous execution complete";
      throw new RuntimeException(errMsg);
    }
    messagePool[idx] = msg;
    messageSignalPool[idx].set(true);
  }

  public void resetMessageSignal(int idx) {
    assert messageSignalPool[idx].get();
    messageSignalPool[idx].set(false);
  }
}
