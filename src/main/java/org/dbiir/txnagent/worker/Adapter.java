package org.dbiir.txnagent.worker;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.common.types.CCType;
import org.dbiir.txnagent.execution.WorkloadConfiguration;

public class Adapter {
  private static final Adapter INSTANCE;
  private CCType currentCCType = CCType.SER;
  @Getter private CCType nextCCType;
  @Getter private CCType transitionLockCCType = CCType.NUM_CC;
  private final List<OnlineWorker> workers;
  @Setter private WorkloadConfiguration workConf;
  @Getter private boolean inSwitchPhase;
  @Getter private boolean allWorkersReadyForSwitch;
  private final Lock guardForWorkers;

  static {
    INSTANCE = new Adapter();
  }

  public Adapter() {
    this.guardForWorkers = new ReentrantLock();
    this.workers = new ArrayList<>(128);
  }

  public CCType getCCType() {
    if (inSwitchPhase) return transitionLockCCType;
    else return currentCCType;
  }

  public static Adapter getInstance() {
    return INSTANCE;
  }

  public void addOnlineWorker(OnlineWorker worker) {
    guardForWorkers.lock();
    this.workers.add(worker);
    guardForWorkers.unlock();
  }

  public void removeOnlineWorker(int id) {
    guardForWorkers.lock();
    workers.removeIf(w -> w.getId() == id);
    guardForWorkers.unlock();
  }

  public void setNextCCType(CCType ccType) {
    if (workConf.getConcurrencyControlType() != CCType.DYNAMIC) {
      return;
    }
    nextCCType = ccType;
    if (nextCCType == currentCCType) {
      nextCCType = CCType.NUM_CC;
    } else {
      /*
       * 1. (not necessary) set the isolation level in benchmark module
       * 2. notify all workers the new isolation to block the transaction that no enters validation: transition phase
       * 3. travel through the worker list to determine whether all workers finish the switch
       * 4. notify all workers to use the validation in new isolation level
       */
      System.out.println("=====*===== Switch from " + currentCCType + " to " + nextCCType);
      long startTime = System.currentTimeMillis();
      setTransitionLockCCType();
      inSwitchPhase = true;
      while (!allSwitchPhaseReady() && !Thread.interrupted()) {
        try {
          Thread.sleep(1);
        } catch (InterruptedException e) {
          System.out.println("Thread interrupted");
        }
      }
      allWorkersReadyForSwitch = true;
      long endTime = System.currentTimeMillis();
      System.out.println(
          "=====*===== All workers ready for transition" + (endTime - startTime) + "ms");
      while (!allSwitchFinish() && !Thread.interrupted()) {
        try {
          Thread.sleep(1);
        } catch (InterruptedException e) {
          System.out.println("Thread interrupted");
        }
      }
      currentCCType = nextCCType;
      inSwitchPhase = false;
      allWorkersReadyForSwitch = false;
      resetWorkerStatus();
      endTime = System.currentTimeMillis();
      System.out.println(
          "=====*===== All workers finish transition" + (endTime - startTime) + "ms");
      endTime = System.currentTimeMillis();
      System.out.println("=====*===== Switch phase takes " + (endTime - startTime) + "ms");
      nextCCType = CCType.NUM_CC;
    }
  }

  public void setNextCCType(String ccType) {
    ccType = ccType.trim();
    if (ccType.equals("0")) {
      this.setNextCCType(CCType.SER);
    } else if (ccType.equals("1")) {
      this.setNextCCType(CCType.SI_TAILOR);
    } else if (ccType.equals("2")) {
      this.setNextCCType(CCType.RC_TAILOR);
    } else {
      System.out.println("ERROR");
    }
  }

  private boolean allSwitchFinish() {
    guardForWorkers.lock();
    for (OnlineWorker worker : workers) {
      if (!worker.isSwitchFinish()) {
        try {
          System.out.println(worker.toString() + " is not finish");
          Thread.sleep(5);
        } catch (InterruptedException e) {
          guardForWorkers.unlock();
          System.out.println("Thread.sleep() interrupted");
        }
        guardForWorkers.unlock();
        return false;
      }
    }
    guardForWorkers.unlock();
    return true;
  }

  private void resetWorkerStatus() {
    for (OnlineWorker worker : workers) {
      worker.setSwitchPhaseReady(false);
      worker.setSwitchFinish(false);
    }
  }

  private boolean allSwitchPhaseReady() {
    // ready if its validation is
    guardForWorkers.lock();
    for (OnlineWorker worker : workers) {
      if (!worker.isSwitchPhaseReady()) {
        try {
          System.out.println(worker.toString() + " is not ready");
          Thread.sleep(5);
        } catch (InterruptedException e) {
          guardForWorkers.unlock();
        }
        guardForWorkers.unlock();
        return false;
      }
    }
    guardForWorkers.unlock();
    return true;
  }

  // set out the lock manner in switch phase
  private void setTransitionLockCCType() {
    if (underRC(currentCCType) || underRC(nextCCType)) {
      this.transitionLockCCType = CCType.RC_TAILOR;
    } else if (underSI(currentCCType) || underSI(nextCCType)) {
      this.transitionLockCCType = CCType.SI_TAILOR;
    } else {
      this.transitionLockCCType = CCType.SER;
    }
  }

  private boolean underRC(CCType type) {
    return type == CCType.RC || type == CCType.RC_TAILOR;
  }

  private boolean underSI(CCType type) {
    return type == CCType.SI || type == CCType.SI_TAILOR;
  }
}
