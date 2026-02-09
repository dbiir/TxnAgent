package org.dbiir.txnagent.execution.isolation;

import java.io.IOException;
import java.net.Socket;
import lombok.Setter;
import org.dbiir.txnagent.common.types.CCType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class StatisticsThread extends Thread {
  private static final Logger logger = LoggerFactory.getLogger(StatisticsThread.class);
  private static final String ip = "localhost";
  private static final int port = 8765;
  @Setter private String outputFilePrefix;
  private CCType ccType;
  private Socket socket;

  public StatisticsThread(String prefix, CCType ccType) {
    this.outputFilePrefix = prefix;
    this.ccType = ccType;
    try {
      this.socket = new Socket(ip, port);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void run() {
    while (!Thread.interrupted()) {
      try {
        // TODO: collect the statistics of data items
        Thread.sleep(5000L);
      } catch (InterruptedException e) {
        logger.info("Stastic thread interrupted: {}", e.getMessage());
      }
    }
  }
}
