package org.dbiir.txnsails.execution.isolation;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.dbiir.txnsails.common.AsyncResultWrapper;
import org.dbiir.txnsails.common.TransactionStatus;
import org.dbiir.txnsails.common.ValidationStatus;
import org.dbiir.txnsails.execution.validation.ValidationMeta;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TransactionManager {
  private static final Logger logger = LoggerFactory.getLogger(TransactionManager.class);
  private static final TransactionManager INSTANCE = new TransactionManager();
  private static final int TRANSACTION_HASH_SIZE = 128; // must be configured to 2^n

  private String workload;
  // active transaction list
  private final List<List<Transaction>> activeTransactionList =
          new ArrayList<>(TRANSACTION_HASH_SIZE);
  private final List<ReadWriteLock> activeTransactionLocks = new ArrayList<>(TRANSACTION_HASH_SIZE);

  // data item metadata, including hot, medium, low contention items

  public TransactionManager() {
    for (int i = 0; i < TRANSACTION_HASH_SIZE; i++) {
      activeTransactionList.add(new LinkedList<>());
      activeTransactionLocks.add(new ReentrantReadWriteLock());
    }
  }

  public void init(String workload) {
    this.workload = workload;
  }

  public void read(Transaction transaction, ValidationMeta validationMeta) {
    DataItem item = PartitionManager.getInstance().getAndAddDataItem(validationMeta);
    item.read(transaction, validationMeta.getOldVersion());
    validationMeta.setDataItem(item);
  }

  public void write(Transaction transaction, ValidationMeta validationMeta) {
    DataItem item = PartitionManager.getInstance().getAndAddDataItem(validationMeta);
    item.write(transaction);
    validationMeta.setDataItem(item);
  }

  public void prepare(Transaction transaction) throws SQLException {
    boolean hybridTransaction = transaction.getParticipants().size() > 1;

    for (Participant p : transaction.getParticipants()) {
      //      p.validate(transaction);
      p.setValidationStatus(ValidationStatus.VALIDATED);

      // serial execution
      if (hybridTransaction) {
        try (PreparedStatement prepare =
                     p.getConnection()
                             .prepareStatement(
                                     "PREPARE TRANSACTION '"
                                             + transaction.getId()
                                             + "-"
                                             + p.getIsolationLevel()
                                             + "'"); ) {
          prepare.execute();
          p.setStatus(TransactionStatus.PREPARED);
        } catch (SQLException ex) {
          p.setStatus(TransactionStatus.PREPARE_FAILED);
          throw ex;
        }
      }
    }
  }

  public void commit(Transaction transaction, AsyncResultWrapper[] results) {
    for (Participant p : transaction.getParticipants()) {
      if (p.getValidationStatus() != ValidationStatus.VALIDATED) {
        logger.error(
                Thread.currentThread().getName()
                        + " The transaction is not validated !!!"
                        + p.getValidationStatus());
        return;
      }
      if (p.getConnection() != null) {
        try {
          if (p.getStatus() == TransactionStatus.PREPARED) {
            PreparedStatement prepare =
                    p.getConnection()
                            .prepareStatement(
                                    "COMMIT PREPARED '"
                                            + transaction.getId()
                                            + "-"
                                            + p.getIsolationLevel()
                                            + "'");
            prepare.execute();
          } else if (p.getStatus() == TransactionStatus.ACTIVE) {
            PreparedStatement prepare = p.getConnection().prepareStatement("COMMIT");
            prepare.execute();
          }
        } catch (SQLException ex) {
          logger.error("Commit failed !!!", ex);
          results[p.getIsolationLevel()].setException(ex);
        }
      } else {
        logger.error("Cannot find the connection");
      }

      //      p.doAfterCommit(transaction);
    }
  }

  public void rollback(Transaction transaction, AsyncResultWrapper[] results) {
    for (Participant p : transaction.getParticipants()) {
      if (p.getConnection() != null) {
        try {
          if (p.getStatus() == TransactionStatus.PREPARED) {
            PreparedStatement prepare =
                    p.getConnection()
                            .prepareStatement(
                                    "ROLLBACK PREPARED '"
                                            + transaction.getId()
                                            + "-"
                                            + p.getIsolationLevel()
                                            + "'");
            prepare.execute();
          } else if (p.getStatus() == TransactionStatus.PREPARE_FAILED
                  || p.getStatus() == TransactionStatus.ACTIVE) {
            PreparedStatement prepare = p.getConnection().prepareStatement("ROLLBACK");
            prepare.execute();
          } else {
            logger.error(
                    "The transaction status is {}; Validation status is {}",
                    p.getStatus(),
                    p.getValidationStatus());
          }
          results[p.getIsolationLevel()].setException(null);
        } catch (SQLException ex) {
          results[p.getIsolationLevel()].setException(ex);
        }
      } else {
        logger.error("Cannot find the connection");
      }

      //      p.doAfterRollback(transaction);
    }
  }

  public void addTransaction(Transaction txn) {
    int hash = (int) (txn.getId() & (TRANSACTION_HASH_SIZE - 1));
    activeTransactionLocks.get(hash).writeLock().lock();
    activeTransactionList.get(hash).add(txn);
    activeTransactionLocks.get(hash).writeLock().unlock();
  }

  public void removeTransaction(Transaction txn) {
    int hash = (int) (txn.getId() & (TRANSACTION_HASH_SIZE - 1));
    activeTransactionLocks.get(hash).writeLock().lock();
    Iterator<Transaction> iterator = activeTransactionList.get(hash).iterator();
    while (iterator.hasNext()) {
      Transaction t = iterator.next();
      if (t.getId() == txn.getId()) {
        iterator.remove();
        break;
      }
    }
    activeTransactionLocks.get(hash).writeLock().unlock();
  }

  public Transaction getTransaction(long tid) {
    int hash = (int) (tid & (TRANSACTION_HASH_SIZE - 1));
    activeTransactionLocks.get(hash).readLock().lock();
    List<Transaction> txnList = activeTransactionList.get(hash);
    for (Transaction txn : txnList) {
      if (txn.getId() == tid) {
        activeTransactionLocks.get(hash).readLock().unlock();
        return txn;
      }
    }
    activeTransactionLocks.get(hash).readLock().unlock();
    return null;
  }

  public long getMinActiveTransactionId() {
    long minTid = Long.MAX_VALUE;
    for (int i = 0; i < TRANSACTION_HASH_SIZE; i++) {
      activeTransactionLocks.get(i).readLock().lock();
      List<Transaction> txnList = activeTransactionList.get(i);
      for (Transaction txn : txnList) {
        minTid = Math.min(minTid, txn.getId());
      }
      activeTransactionLocks.get(i).readLock().unlock();
    }
    return minTid;
  }

  public static TransactionManager getInstance() {
    return INSTANCE;
  }
}
