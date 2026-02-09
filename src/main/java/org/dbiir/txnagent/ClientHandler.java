package org.dbiir.txnagent;

import java.io.*;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.sql.SQLException;
import java.text.MessageFormat;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.dbiir.txnagent.common.types.CCType;
import org.dbiir.txnagent.execution.WorkloadConfiguration;
import org.dbiir.txnagent.worker.MetaWorker;
import org.dbiir.txnagent.worker.OfflineWorker;
import org.dbiir.txnagent.worker.OnlineWorker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class ClientHandler implements Runnable {
  private static final Logger logger = LoggerFactory.getLogger(ClientHandler.class);
  private static final int BUFFER_SIZE = 4096;
  private static final boolean asyncCommit = false;
  private final Socket clientSocket;
  private final WorkloadConfiguration configuration;
  private final int id;
  private final OnlineWorker worker;
  private final CCType ccType;
  ExecutorService executor = Executors.newFixedThreadPool(1);
  Future<?> future;

  public ClientHandler(Socket clientSocket, WorkloadConfiguration configuration, int id) {
    this.clientSocket = clientSocket;
    this.configuration = configuration;
    this.id = id;
    this.worker = new OnlineWorker(configuration, id);
    this.ccType = configuration.getConcurrencyControlType();
  }

  @Override
  public void run() {
    try (BufferedReader in =
            new BufferedReader(
                new InputStreamReader(clientSocket.getInputStream(), StandardCharsets.UTF_8),
                BUFFER_SIZE);
        BufferedWriter out =
            new BufferedWriter(
                new OutputStreamWriter(clientSocket.getOutputStream(), StandardCharsets.UTF_8),
                BUFFER_SIZE)) {
      String message;
      while ((message = in.readLine()) != null) {
        long start = System.nanoTime();
        String[] queries = parseQueries(message.trim());
        String response = "";
        for (String q : queries) {
          // logger.debug("{} Received: {}", worker.toString(), q);
          String[] args = parseArgs(q.trim());
          String functionName = args[0].toLowerCase();
          try {
            switch (functionName) {
              case "execute" -> {
                if (ccType == CCType.FS) {
                  response = worker.executeFS(args, 3);
                  if (worker.isLastSQL()) {
                    if (asyncCommit) {
                      future =
                          executor.submit(
                              () -> {
                                try {
                                  worker.prepare();
                                } catch (SQLException e) {
                                  logger.error(e.getMessage());
                                }
                              });
                    } else {
                      worker.prepare();
                    }
                  }
                } else {
                  response = worker.execute(args, 3);
                }
                response = "OK#" + response;
              }
              case "commit" -> {
                if (ccType == CCType.FS) {
                  if (asyncCommit) {
                    future.get();
                    future =
                        executor.submit(
                            () -> {
                              try {
                                worker.commitFS();
                              } catch (Exception e) {
                                e.printStackTrace();
                              }
                            });
                  } else {
                    worker.commitFS();
                  }
                } else {
                  worker.commit();
                }
                response = "OK";
              }
              case "rollback" -> {
                if (ccType == CCType.FS) {
                  worker.rollbackFS();
                } else {
                  worker.rollback();
                }
                response = "OK";
              }
              case "register" -> {
                if (args.length < 4) {
                  response = "FAILED";
                  break;
                }
                int idx = OfflineWorker.getINSTANCE().register(args);
                if (idx < 0) {
                  response = "FAILED";
                } else {
                  response = "OK#" + idx; // response with the unique sql index in server-side
                }
              }
              case "analysis" -> {
                response = "OK";
                OfflineWorker.getINSTANCE().register_end(args);
              }
              case "close" -> {
                clientSocket.close();
                TxnSailsServer.closeServer();
                return;
              }
              default -> {
                response = "Unknown function: " + functionName;
              }
            }
          } catch (SQLException ex) {
            response =
                MessageFormat.format(
                    MetaWorker.ERROR_FORMATTER,
                    ex.getMessage().split("\n")[0],
                    ex.getSQLState(),
                    ex.getErrorCode());
            break;
          }
        }
        logger.debug("Execution time: {}us", (System.nanoTime() - start) / 1000);
        // logger.debug("{} response: {}", worker.toString(), response);
        out.write(response + "\n");
        out.flush();
      }
    } catch (IOException ex) {
      System.out.println("Client disconnected: " + id);
    } catch (Exception e) {
      System.out.println(List.of(e.getStackTrace()));
    } finally {
      try {
        clientSocket.close();
        worker.closeWorker();
      } catch (Exception e) {
        System.out.println(List.of(e.getStackTrace()));
      }
    }
  }

  private String[] parseArgs(String message) {
    String[] parts = message.split("#");
    for (int i = 0; i < parts.length; i++) {
      parts[i] = parts[i].trim();
    }
    return parts;
  }

  private String[] parseQueries(String message) {
    String[] queries = message.split("@");
    for (int i = 0; i < queries.length; i++) {
      queries[i] = queries[i].trim();
    }
    return queries;
  }

  private void sendResponse(PrintWriter out, String response) {
    System.out.println("Sending: " + response);
    out.println(response);
  }
}
