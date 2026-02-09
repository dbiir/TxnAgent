package org.dbiir.txnagent.common;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;

public interface CLibrary extends Library {
  CLibrary INSTANCE = Native.load(Platform.isWindows() ? "msvcrt" : "c", CLibrary.class);

  int sched_setaffinity(int pid, int cpusetsize, Pointer cpuset);

  int sched_getaffinity(int pid, int cpusetsize, Pointer cpuset);
}
