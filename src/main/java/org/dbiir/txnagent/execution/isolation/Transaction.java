package org.dbiir.txnagent.execution.isolation;

import java.util.LinkedList;
import java.util.List;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.common.TransactionStatus;
import org.dbiir.txnagent.execution.utils.SpinLock;

public class Transaction {
  @Getter private long id;
  @Getter private final List<Participant> participants;
  @Getter @Setter private long lowerBound;
  @Getter @Setter private long upperBound;
  @Getter @Setter private TransactionStatus status;
  private final SpinLock lock;
  @Getter @Setter private long commitTimestamp;

  public Transaction(long tid, List<Participant> participants) {
    this.id = tid;
    this.participants = participants;
    this.lowerBound = System.nanoTime();
    this.upperBound = Long.MAX_VALUE;
    this.lock = new SpinLock();
    this.status = TransactionStatus.IDLE;
  }

  public Transaction(long tid) {
    this.id = tid;
    this.participants = new LinkedList<>();
    this.lowerBound = System.nanoTime();
    this.upperBound = Long.MAX_VALUE;
    this.status = TransactionStatus.IDLE;
    this.lock = new SpinLock();
  }

  public void addParticipant(Participant participant) {
    this.participants.add(participant);
  }

  // reset the transaction id and clear participants
  public void init(long tid) {
    this.id = tid;
    this.status = TransactionStatus.ACTIVE;
    this.lowerBound = System.nanoTime();
    this.upperBound = Long.MAX_VALUE;
  }

  public void reset() {
    this.participants.clear();
    this.status = TransactionStatus.IDLE;
  }

  public void spinLock() {
    this.lock.lock();
  }

  public void spinUnlock() {
    this.lock.unlock();
  }

  public boolean readOnly() {
    for (Participant participant : participants) {
      if (!participant.readOnly()) {
        return false;
      }
    }
    return true;
  }

  public void doAfterCommit(long minActiveTransactionId) {
    this.status = TransactionStatus.COMMITTED;
    for (Participant participant : this.participants) {
      // write set
      int itemCount = participant.getWriteSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        DataItem dataItem = participant.getWriteSet().get(i).getDataItem();
        dataItem.installVersion(
            participant.getWriteSet().get(i).getOldVersion(),
            commitTimestamp,
            minActiveTransactionId);
        dataItem.setMaxReadTimestamp(commitTimestamp);
      }

      // read set
      itemCount = participant.getReadSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        participant.getReadSet().get(i).getDataItem().removeReadTransaction(this);
      }
    }
  }

  public void doAfterRollback() {
    for (Participant participant : participants) {
      // read set
      int itemCount = participant.getReadSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        // atomic remove transaction from read transaction list
        participant.getReadSet().get(i).getDataItem().removeReadTransaction(this);
      }
    }
  }

  public String getReadSetString() {
    StringBuilder sb = new StringBuilder();
    for (Participant participant : participants) {
      int itemCount = participant.getReadSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        DataItem dataItem = participant.getReadSet().get(i).getDataItem();
        sb.append(dataItem.getKey()).append(", ");
      }
    }
    return sb.toString().trim();
  }

  public String getWriteSetString() {
    StringBuilder sb = new StringBuilder();
    for (Participant participant : participants) {
      int itemCount = participant.getWriteSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        DataItem dataItem = participant.getWriteSet().get(i).getDataItem();
        sb.append(dataItem.getKey()).append(", ");
      }
    }
    return sb.toString().trim();
  }
}
