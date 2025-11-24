package org.dbiir.txnsails.execution.isolation;

import java.util.LinkedList;
import java.util.List;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnsails.common.TransactionStatus;
import org.dbiir.txnsails.execution.utils.SpinLock;

public class Transaction {
  @Getter private long id;
  @Getter private final List<Participant> participants;
  @Getter @Setter private long lowerBound;
  @Getter @Setter private long upperBound;
  @Getter @Setter private boolean prepared;
  @Getter @Setter private TransactionStatus status;
  private final SpinLock lock;
  @Getter @Setter private long commitTimestamp;

  public Transaction(long tid, List<Participant> participants) {
    this.id = tid;
    this.participants = participants;
    this.lowerBound = System.currentTimeMillis();
    this.upperBound = Long.MAX_VALUE;
    this.lock = new SpinLock();
    this.status = TransactionStatus.ACTIVE;
  }

  public Transaction(long tid) {
    this.id = tid;
    this.participants = new LinkedList<>();
    this.lowerBound = System.currentTimeMillis();
    this.upperBound = Long.MAX_VALUE;
    this.lock = new SpinLock();
  }

  public void addParticipant(Participant participant) {
    this.participants.add(participant);
  }

  // reset the transaction id and clear participants
  public void init(long tid) {
    this.id = tid;
    this.lowerBound = System.currentTimeMillis();
    this.upperBound = Long.MAX_VALUE;
  }

  public void reset() {
    this.participants.clear();
    this.status = TransactionStatus.ACTIVE;
  }

  public void spinLock() {
    this.lock.lock();
  }

  public void spinUnlock() {
    this.lock.unlock();
  }
}
