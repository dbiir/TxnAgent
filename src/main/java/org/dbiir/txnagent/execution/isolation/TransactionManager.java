package org.dbiir.txnagent.execution.isolation;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import org.dbiir.txnagent.common.AsyncResultWrapper;
import org.dbiir.txnagent.common.TransactionStatus;
import org.dbiir.txnagent.common.ValidationStatus;
import org.dbiir.txnagent.execution.validation.ValidationMeta;
import org.dbiir.txnagent.worker.StatisticsWorker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TransactionManager {
  private static final Logger logger = LoggerFactory.getLogger(TransactionManager.class);
  private static final TransactionManager INSTANCE = new TransactionManager();
  private static final int TRANSACTION_HASH_SIZE = 128; // must be configured to 2^n

  private String workload;
  private StatisticsWorker statisticsWorker;
  // active transaction list
  private final List<List<Transaction>> activeTransactionList = new ArrayList<>(TRANSACTION_HASH_SIZE);
  private final List<ReadWriteLock> activeTransactionLocks = new ArrayList<>(TRANSACTION_HASH_SIZE);

  // data item metadata, including hot, medium, low contention items

  public TransactionManager() {
    for (int i = 0; i < TRANSACTION_HASH_SIZE; i++) {
      activeTransactionList.add(new LinkedList<>());
      activeTransactionLocks.add(new ReentrantReadWriteLock());
    }
  }

  public void init(String workload, StatisticsWorker statisticsWorker) {
    this.workload = workload;
    this.statisticsWorker = statisticsWorker;
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
    boolean readOnly = transaction.readOnly();

    // Phase 1: validate all participants
    for (Participant p : transaction.getParticipants()) {
      p.validate(transaction);
      p.setValidationStatus(ValidationStatus.VALIDATED);
    }

    // Phase 2: send all PREPARE TRANSACTION statements in parallel
    if (hybridTransaction && !readOnly) {
      List<Participant> participants = transaction.getParticipants();
      AtomicReference<SQLException> firstException = new AtomicReference<>();
      List<CompletableFuture<Void>> futures = new ArrayList<>(participants.size() - 1);

      // Submit participants 1..n-1 to async pool
      for (int i = 1; i < participants.size(); i++) {
        Participant p = participants.get(i);
        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
          try (PreparedStatement stmt = p.getConnection()
              .prepareStatement(
                  "PREPARE TRANSACTION '"
                      + transaction.getId()
                      + "-"
                      + p.getIsolationLevel()
                      + "'")) {
            stmt.execute();
            p.setStatus(TransactionStatus.PREPARED);
          } catch (SQLException ex) {
            p.setStatus(TransactionStatus.PREPARE_FAILED);
            firstException.compareAndSet(null, ex);
          }
        });
        futures.add(future);
      }

      // Current thread executes participant 0
      Participant p0 = participants.get(0);
      try (PreparedStatement stmt = p0.getConnection()
          .prepareStatement(
              "PREPARE TRANSACTION '"
                  + transaction.getId()
                  + "-"
                  + p0.getIsolationLevel()
                  + "'")) {
        stmt.execute();
        p0.setStatus(TransactionStatus.PREPARED);
      } catch (SQLException ex) {
        p0.setStatus(TransactionStatus.PREPARE_FAILED);
        firstException.compareAndSet(null, ex);
      }

      // Wait for async branches to complete
      CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

      if (firstException.get() != null) {
        throw firstException.get();
      }
    }

    transaction.setCommitTimestamp(transaction.getLowerBound());
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
            PreparedStatement prepare = p.getConnection()
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
          transaction.doAfterCommit(getMinActiveTransactionId());
          statisticsWorker.recordTransaction(transaction, true);
        } catch (SQLException ex) {
          if (p.getStatus() == TransactionStatus.PREPARED) {
            logger.error("Commit failed !!!", ex);
          }
          results[p.getIsolationLevel()].setException(ex);
        }
      } else {
        logger.error("Cannot find the connection");
      }
    }
  }

  public void rollback(Transaction transaction, AsyncResultWrapper[] results) {
    for (Participant p : transaction.getParticipants()) {
      if (p.getConnection() != null) {
        try {
          if (p.getStatus() == TransactionStatus.PREPARED) {
            PreparedStatement prepare = p.getConnection()
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
    }
    transaction.doAfterRollback();
    statisticsWorker.recordTransaction(transaction, false);
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
