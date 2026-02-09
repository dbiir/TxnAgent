package org.dbiir.txnagent.common;

import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.common.types.DependencyType;

@Getter
public class StaticDependencyGraphEdge {
  private final TransactionTemplate from;
  private final TransactionTemplate to;
  private final DependencyType type;

  /**
   * edgeName format is "name1-name2-idx1-idx2" case1: READ-WRITE or WRITE-READ - name1 is the
   * `read`, id1 is the index in corresponding template - name2 is the `write`, id2 is the index in
   * corresponding template case2: WRITE-WRITE - name1 and name2 in any consistent sequence
   */
  @Setter private String edgeName = "";

  @Setter private boolean hasKeptOrder = false;
  @Setter private int idxInFrom;
  @Setter private int idxInTo;

  public StaticDependencyGraphEdge(
      TransactionTemplate from, TransactionTemplate to, DependencyType type) {
    this.from = from;
    this.to = to;
    this.type = type;
  }

  @Override
  public String toString() {
    return "DependencyEdge{"
        + "from="
        + from.getName()
        + ", to="
        + to.getName()
        + ", type="
        + type
        + '}';
  }
}
