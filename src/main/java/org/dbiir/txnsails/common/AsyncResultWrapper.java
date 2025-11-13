package org.dbiir.txnsails.common;

import lombok.Getter;
import lombok.Setter;
import org.dbiir.txnsails.common.types.IsolationLevelType;

import java.sql.Connection;

@Getter
public class AsyncResultWrapper {
  private Connection connection;
  private IsolationLevelType isolationLevel;
  @Setter private Exception exception;

  public AsyncResultWrapper(Connection connection, IsolationLevelType isolationLevelType, Exception exception) {
    this.connection = connection;
    this.isolationLevel = isolationLevelType;
    this.exception = exception;
  }

  public AsyncResultWrapper(Connection connection, IsolationLevelType isolationLevelType) {
    this.connection = connection;
    this.isolationLevel = isolationLevelType;
    this.exception = null;
  }

  public boolean isSuccess() {
    return exception == null;
  }

  public void reset() {
    this.exception = null;
  }
}

