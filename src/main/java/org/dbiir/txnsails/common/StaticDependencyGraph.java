package org.dbiir.txnsails.common;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import lombok.Getter;
import org.dbiir.txnsails.common.types.DependencyType;

public class StaticDependencyGraph {
  @Getter private HashMap<TransactionTemplate, List<StaticDependencyGraphEdge>> adjacencyList;
  private final List<TransactionTemplate> templateList = new LinkedList<>();

  public StaticDependencyGraph() {
    this.adjacencyList = new HashMap<>();
  }

  public void addTemplate(TransactionTemplate template) {
    if (adjacencyList.containsKey(template)) return;
    adjacencyList.putIfAbsent(template, new ArrayList<>());
    templateList.add(template);
  }

  public void addDependency(TransactionTemplate from, TransactionTemplate to, DependencyType type) {
    StaticDependencyGraphEdge edge = new StaticDependencyGraphEdge(from, to, type);
    adjacencyList.get(from).add(edge);
  }

  public void addDependency(
      TransactionTemplate from, TransactionTemplate to, DependencyType type, String edgeName) {
    StaticDependencyGraphEdge edge = new StaticDependencyGraphEdge(from, to, type);
    edge.setEdgeName(edgeName);
    adjacencyList.get(from).add(edge);
  }

  public void addDependency(
      TransactionTemplate from,
      TransactionTemplate to,
      DependencyType type,
      String edgeName,
      int idxInFrom,
      int idxInTo) {
    StaticDependencyGraphEdge edge = new StaticDependencyGraphEdge(from, to, type);
    edge.setEdgeName(edgeName);
    edge.setIdxInFrom(idxInFrom);
    edge.setIdxInTo(idxInTo);
    adjacencyList.get(from).add(edge);
    System.out.println(
        "Edge: {Type: " + type + ", From: " + from.getName() + ", To: " + to.getName() + "}");
  }

  public List<StaticDependencyGraphEdge> getDependencies(TransactionTemplate template) {
    return adjacencyList.get(template);
  }

  public void addAllDependenciesByTemplates() {
    int listLength = templateList.size();
    List<TransactionTemplate> templateArrayList = new ArrayList<>();
    int idx = 0;
    for (TransactionTemplate t : templateList) {
      templateArrayList.add(t);
      idx++;
    }

    for (int i = 0; i < listLength; i++) {
      for (int j = i + 1; j < listLength; j++) {
        TransactionTemplate t1 = templateArrayList.get(i);
        TransactionTemplate t2 = templateArrayList.get(j);
        // find the dependencies
        addDependency(t1, t2);
      }
    }
  }

  private void removeDependency(List<StaticDependencyGraphEdge> edges, String edgeName) {
    for (StaticDependencyGraphEdge edge : edges) {
      if (edge.getEdgeName().equals(edgeName)) {
        edges.remove(edge);
        return;
      }
    }
  }

  private void addDependency(TransactionTemplate t1, TransactionTemplate t2) {
    int total_t1 = t1.totalSQLs();
    int total_t2 = t2.totalSQLs();
    boolean hasRW = false;
    boolean hasWW = false;
    for (int i = 0; i < total_t1; i++) {
      if (t1.shouldSkipIndex(i)) {
        continue;
      }
      for (int j = 0; j < total_t2; j++) {
        if (t2.shouldSkipIndex(j)) {
          continue;
        }
        if (t1.getRelationByIndex(i).equals(t2.getRelationByIndex(j))) {
          // 0 represents read, 1 represents write
          if (t1.getOperationByIndex(i) == 0 && t2.getOperationByIndex(j) == 1) {
            String edge_name = t1.getName() + "-" + t2.getName() + "-" + i + "-" + j;
            addDependency(t1, t2, DependencyType.READ_WRITE, edge_name, i, j);
            addDependency(t2, t1, DependencyType.WRITE_READ, edge_name, j, i);
            hasRW = true;
          } else if (t1.getOperationByIndex(i) == 1 && t2.getOperationByIndex(j) == 0) {
            String edge_name = t2.getName() + "-" + t1.getName() + "-" + j + "-" + i;
            addDependency(t2, t1, DependencyType.READ_WRITE, edge_name, j, i);
            addDependency(t1, t2, DependencyType.WRITE_READ, edge_name, i, j);
            hasRW = true;
          } else if (t1.getOperationByIndex(i) == 1 && t2.getOperationByIndex(j) == 1) {
            String edge_name = t1.getName() + "-" + t2.getName() + "-" + i + "-" + j;
            addDependency(t1, t2, DependencyType.WRITE_WRITE, edge_name, i, j);
            addDependency(t2, t1, DependencyType.WRITE_WRITE, edge_name, j, i);
            hasWW = true;
          }
        }
      }
    }

    // remove unnecessary RW dependencies because of the following WW dependencies
    if (hasWW && hasRW) {
      boolean flag = true;
      while (flag) {
        List<StaticDependencyGraphEdge> t1_edges = adjacencyList.get(t1);
        List<StaticDependencyGraphEdge> t2_edges = adjacencyList.get(t2);
        for (int i = 0; i < t1_edges.size(); i++) {
          StaticDependencyGraphEdge edge = t1_edges.get(i);
          if (edge.getTo().equals(t2) && edge.getType() == DependencyType.READ_WRITE) {
            System.out.println(t1.getName() + " remove RW dependency: " + edge.getEdgeName());
            removeDependency(t1_edges, edge.getEdgeName());
            removeDependency(t2_edges, edge.getEdgeName());
            break;
          } else if (edge.getTo().equals(t2) && edge.getType() == DependencyType.WRITE_READ) {
            System.out.println(t1.getName() + " remove WR dependency: " + edge.getEdgeName());
            removeDependency(t1_edges, edge.getEdgeName());
            removeDependency(t2_edges, edge.getEdgeName());
            break;
          }
        }
        flag = false;
      }
    }
    return;
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (HashMap.Entry<TransactionTemplate, List<StaticDependencyGraphEdge>> entry :
        adjacencyList.entrySet()) {
      sb.append(entry.getKey().toString())
          .append(" -> ")
          .append(entry.getValue().toString())
          .append("\n");
    }
    return sb.toString();
  }
}
