package org.dbiir.txnagent.common;

import java.sql.Connection;
import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnagent.common.types.IsolationLevelType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Getter
public class AsyncResultWrapper {
  private static final Logger logger = LoggerFactory.getLogger(AsyncResultWrapper.class);
  @Setter private Connection connection;
  private IsolationLevelType isolationLevel;
  @Setter private Exception exception;

  public AsyncResultWrapper(
      Connection connection, IsolationLevelType isolationLevelType, Exception exception) {
    this.connection = connection;
    this.isolationLevel = isolationLevelType;
    this.exception = exception;
  }

  public AsyncResultWrapper(Connection connection, IsolationLevelType isolationLevelType) {
    this.connection = connection;
    this.isolationLevel = isolationLevelType;
    this.exception = null;
  }

  public AsyncResultWrapper(Connection connection, int isolationLevelType) {
    this.connection = connection;
    switch (isolationLevelType) {
      case 0 -> {
        this.isolationLevel = IsolationLevelType.SER;
      }
      case 1 -> {
        this.isolationLevel = IsolationLevelType.SI;
      }
      case 2 -> {
        this.isolationLevel = IsolationLevelType.RC;
      }
      default -> {
        logger.error("");
      }
    }
    this.exception = null;
  }

  public boolean isSuccess() {
    return exception == null;
  }

  public void reset() {
    this.exception = null;
  }
}
