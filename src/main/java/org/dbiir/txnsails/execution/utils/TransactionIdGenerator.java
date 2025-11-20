package org.dbiir.txnsails.execution.utils;

public class TransactionIdGenerator {
  private static final int ID_BITS = 8; // (0-255)
  private static final int THREAD_BITS = 10; // (0-1023)
  private static final int TIMESTAMP_BITS = 46; //

  // 移位常量
  private static final int THREAD_ID_SHIFT = ID_BITS; // 8位
  private static final int TIMESTAMP_SHIFT = THREAD_BITS + ID_BITS; // 18位

  // 掩码常量
  private static final long ID_MASK = (1L << ID_BITS) - 1; // 0xFF (255)
  private static final long THREAD_ID_MASK = (1L << THREAD_BITS) - 1; // 0x3FF (1023)
  private static final long TIMESTAMP_MASK = (1L << TIMESTAMP_BITS) - 1; // 0x3FFFFFFFFFFFF

  public static long generateTransactionId(int id) {
    validateId(id);

    long timestamp = System.currentTimeMillis();
    long threadId = Thread.currentThread().threadId();

    return ((timestamp & TIMESTAMP_MASK) << TIMESTAMP_SHIFT)
        | ((threadId & THREAD_ID_MASK) << THREAD_ID_SHIFT)
        | (id & ID_MASK);
  }

  public static int extractId(long tid) {
    return (int) (tid & ID_MASK);
  }

  public static long extractThreadId(long tid) {
    return (tid >>> THREAD_ID_SHIFT) & THREAD_ID_MASK;
  }

  public static long extractTimestamp(long tid) {
    return (tid >>> TIMESTAMP_SHIFT) & TIMESTAMP_MASK;
  }

  private static void validateId(int id) {
    if (id < 0 || id > ID_MASK) {
      throw new IllegalArgumentException(String.format("ID must be between 0 and %d", ID_MASK));
    }
  }
}
