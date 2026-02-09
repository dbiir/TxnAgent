package org.dbiir.txnagent.common;

import java.util.List;

public record StaticDependencyCycle(
    List<TransactionTemplate> nodes, List<StaticDependencyGraphEdge> edges) {

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("Cycle: ");
    for (int i = 0; i < nodes.size(); i++) {
      sb.append(nodes.get(i).getName());
      if (i < edges.size()) {
        sb.append(" --(").append(edges.get(i).getType()).append(")--> ");
      }
    }
    return sb.toString();
  }
}
