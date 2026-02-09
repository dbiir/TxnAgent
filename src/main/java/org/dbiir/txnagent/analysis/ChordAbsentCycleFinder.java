package org.dbiir.txnagent.analysis;

import java.util.*;
import org.dbiir.txnagent.common.StaticDependencyCycle;
import org.dbiir.txnagent.common.StaticDependencyGraphEdge;
import org.dbiir.txnagent.common.TransactionTemplate;

public class ChordAbsentCycleFinder {
  private final Map<TransactionTemplate, List<StaticDependencyGraphEdge>> graph;

  public ChordAbsentCycleFinder(Map<TransactionTemplate, List<StaticDependencyGraphEdge>> graph) {
    this.graph = graph;
  }

  public Set<StaticDependencyCycle> findAllCycles() {
    Set<StaticDependencyCycle> cycles = new HashSet<>();
    Set<TransactionTemplate> visited = new HashSet<>();
    Set<TransactionTemplate> stack = new HashSet<>();

    for (TransactionTemplate node : graph.keySet()) {
      dfs(node, visited, stack, new ArrayList<>(), new ArrayList<>(), cycles, new HashSet<>());
    }

    return cycles;
  }

  private void dfs(
      TransactionTemplate node,
      Set<TransactionTemplate> visited,
      Set<TransactionTemplate> stack,
      List<TransactionTemplate> path,
      List<StaticDependencyGraphEdge> edges,
      Set<StaticDependencyCycle> cycles,
      Set<String> edgeNames) {
    visited.add(node);
    stack.add(node);
    path.add(node);

    for (StaticDependencyGraphEdge edge : graph.getOrDefault(node, Collections.emptyList())) {
      TransactionTemplate neighbor = edge.getTo();
      String edgeName = edge.getEdgeName();
      edges.add(edge);

      if (!visited.contains(neighbor)) {
        edgeNames.add(edgeName); // add the edge name
        dfs(neighbor, visited, stack, path, edges, cycles, edgeNames);
        edgeNames.remove(edgeName); // remove the edge name when backtracking
      } else if (stack.contains(neighbor) && !edgeNames.contains(edgeName)) {
        // find a cycle without nodes with the same edgeName
        int index = path.indexOf(neighbor);
        path.add(neighbor);
        List<TransactionTemplate> cycleNodes = new ArrayList<>(path);
        List<StaticDependencyGraphEdge> cycleEdges = new ArrayList<>(edges);

        edgeNames.add(edgeName);
        if (isSimpleCycle(cycleNodes, edgeNames)) {
          // print the cycle
          cycles.add(new StaticDependencyCycle(cycleNodes, cycleEdges));
        }
        edgeNames.remove(edgeName);
        path.remove(path.size() - 1);
      }

      edges.remove(edges.size() - 1);
    }

    visited.remove(node);
    stack.remove(node);
    path.remove(path.size() - 1);
  }

  private boolean isSimpleCycle(List<TransactionTemplate> cycle, Set<String> edgeNames) {
    TransactionTemplate head = cycle.get(0);
    TransactionTemplate tail = cycle.get(cycle.size() - 1);
    if (!head.getName().equals(tail.getName())) return false;
    // check the chord
    Set<TransactionTemplate> cycleSet = new HashSet<>(cycle);
    if (cycleSet.size() == 2) return true;
    for (TransactionTemplate node : cycle.subList(0, cycle.size() - 2)) {
      for (StaticDependencyGraphEdge edge : graph.getOrDefault(node, Collections.emptyList())) {
        String edgeName = edge.getEdgeName();
        if (cycleSet.contains(edge.getTo()) && !edgeNames.contains(edgeName)) {
          //                    System.out.println("Chord found: " + edge.getFrom().getName() + " ->
          // " + edge.getTo().getName() + " (" + edge.getEdgeName() + ")");
          return false;
        }
      }
    }
    //        System.out.print("Head: " + head.getName() + "; Tail: " + tail.getName() + " { ");
    //        for (TransactionTemplate node : cycle) {
    //            System.out.print(" -> " + node.getName());
    //        }
    //        System.out.println(" }");
    return true;
  }
}
