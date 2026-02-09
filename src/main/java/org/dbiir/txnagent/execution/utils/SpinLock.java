package org.dbiir.txnagent.execution.utils;

import java.util.concurrent.atomic.AtomicBoolean;

public class SpinLock {
  private final AtomicBoolean locked = new AtomicBoolean(false);

  public void lock() {
    while (!locked.compareAndSet(false, true)) {
      Thread.yield();
    }
  }

  public void unlock() {
    locked.set(false);
  }
}
