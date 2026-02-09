package org.dbiir.txnagent.execution.isolation;

import java.sql.Connection;
import java.sql.SQLException;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.common.TransactionStatus;
import org.dbiir.txnagent.common.ValidationStatus;
import org.dbiir.txnagent.execution.validation.ValidationMeta;
import org.dbiir.txnagent.execution.validation.ValidationSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Getter
@Setter
public class Participant {
  private static final Logger logger = LoggerFactory.getLogger(Participant.class);
  private static final int MAX_VALIDATION_ITEM_COUNT = 10;
  private int isolationLevel;
  private Connection connection;
  private TransactionStatus status;
  private ValidationStatus validationStatus;
  private ValidationSet readSet;
  private ValidationSet writeSet;

  public Participant(int isolationLevel, Connection connection) {
    this.isolationLevel = isolationLevel;
    this.status = TransactionStatus.ACTIVE;
    this.connection = connection;
    this.readSet = new ValidationSet(MAX_VALIDATION_ITEM_COUNT);
    this.writeSet = new ValidationSet(MAX_VALIDATION_ITEM_COUNT);
    this.validationStatus = ValidationStatus.VALIDATION_STATUS_NUM;
  }

  public Participant(int isolationLevel) {
    this.isolationLevel = isolationLevel;
    this.status = TransactionStatus.ACTIVE;
    this.connection = null;
    this.readSet = new ValidationSet(MAX_VALIDATION_ITEM_COUNT);
    this.writeSet = new ValidationSet(MAX_VALIDATION_ITEM_COUNT);
    this.validationStatus = ValidationStatus.VALIDATION_STATUS_NUM;
  }

  public void addReadValidationMeta(ValidationMeta item) {
    this.readSet.addValidationMeta(item);
  }

  public void addWriteValidationMeta(ValidationMeta item) {
    this.writeSet.addValidationMeta(item);
  }

  public void validate(Transaction transaction) throws SQLException {
    // iterate write set
    int itemCount = writeSet.getItemCount();
    for (int validationPhase = 0; validationPhase < itemCount; validationPhase++) {
      // do timestamp adjustment
      DataItem dataItem = writeSet.get(validationPhase).getDataItem();

      adjustTimestamp(transaction, dataItem);
      transaction.spinLock();
      transaction.setLowerBound(
          Math.max(transaction.getLowerBound(), dataItem.getMaxReadTimestamp() + 1));
      transaction.spinUnlock();

      if (transaction.getLowerBound() > transaction.getUpperBound()) {
        String msg =
            "Transaction "
                + transaction.getId()
                + " failed in validation phase for write set, LB > UB !!!";
        logger.warn(msg);
        validationStatus = ValidationStatus.FAILED;
        throw new SQLException(msg, "500");
      }
    }
    validationStatus = ValidationStatus.VALIDATED;
  }

  public void doAfterCommit(Transaction transaction) {
    for (Participant participant : transaction.getParticipants()) {
      // write set
      int itemCount = participant.getWriteSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        DataItem dataItem = participant.getWriteSet().get(i).getDataItem();
        dataItem.installVersion(
            participant.getWriteSet().get(i).getOldVersion(), transaction.getCommitTimestamp());
        dataItem.setMaxReadTimestamp(transaction.getCommitTimestamp());
        dataItem.releaseWriteLock(transaction.getId());
      }

      // read set
      itemCount = participant.getReadSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        DataItem dataItem = participant.getReadSet().get(i).getDataItem();
        participant.getReadSet().get(i).getDataItem().removeReadTransaction(transaction);
      }
    }
  }

  public void doAfterRollback(Transaction transaction) {
    for (Participant participant : transaction.getParticipants()) {
      // write set
      int itemCount = participant.getWriteSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        DataItem dataItem = participant.getWriteSet().get(i).getDataItem();
        dataItem.setMaxReadTimestamp(transaction.getCommitTimestamp());
        dataItem.releaseWriteLock(transaction.getId());
      }
      // read set
      itemCount = participant.getReadSet().getItemCount();
      for (int i = 0; i < itemCount; i++) {
        // atomic remove transaction from read transaction list
        participant.getReadSet().get(i).getDataItem().removeReadTransaction(transaction);
      }
    }
  }

  private void adjustTimestamp(Transaction t_i, DataItem dataItem) {
    logger.debug("adjustTimestamp: t_i={}, dataItem={}", t_i.getId(), dataItem.getKey());
    dataItem.acquireReadLock();
    for (Transaction t_j : dataItem.getReadTransactions()) {
      logger.debug("adjustTimestamp: t_j={}", t_j.getId());
      // if (t_j.getId() == t_i.getId()) {
      // continue;
      // }
      // add spinlock to concurrent transaction
      if (t_i.getId() < t_j.getId()) {
        t_i.spinLock();
        t_j.spinLock();
      } else {
        t_j.spinLock();
        t_i.spinLock();
      }
      // wait for validated transaction, and continue the validation
      // use RTS to adjust LB
      if (t_j.getStatus().equals(TransactionStatus.VALIDATED)
          || t_j.getStatus().equals(TransactionStatus.PREPARED)) {
        while (t_j.getStatus().equals(TransactionStatus.VALIDATED)
            || t_j.getStatus().equals(TransactionStatus.PREPARED)) {
          Thread.yield();
        }
        continue;
      }

      int mu =
          PartitionManager.getInstance()
              .getMu(dataItem.getPartitionId(), dataItem.getRelationName());
      if (t_i.getLowerBound() < t_j.getLowerBound()) {
        t_i.setLowerBound(t_j.getLowerBound() + mu);
      }
      t_j.setUpperBound(Math.min(t_j.getUpperBound(), t_i.getLowerBound() - 1));

      // release spinlock
      t_i.spinUnlock();
      t_j.spinUnlock();
    }
    dataItem.releaseReadLock();
  }

  public boolean readOnly() {
    return writeSet.isEmpty();
  }

  public void reset() {
    readSet.reset();
    writeSet.reset();
    status = TransactionStatus.ACTIVE;
    validationStatus = ValidationStatus.VALIDATION_STATUS_NUM;
  }
}
