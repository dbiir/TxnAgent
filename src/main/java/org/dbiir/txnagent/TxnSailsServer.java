package org.dbiir.txnagent;

import com.fasterxml.jackson.dataformat.xml.XmlMapper;
import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.sql.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang3.StringUtils;
import org.dbiir.txnagent.analysis.SchemaInfo;
import org.dbiir.txnagent.common.JacksonXmlConfiguration;
import org.dbiir.txnagent.common.types.CCType;
import org.dbiir.txnagent.common.types.DatabaseType;
import org.dbiir.txnagent.execution.WorkloadConfiguration;
import org.dbiir.txnagent.execution.isolation.PartitionManager;
import org.dbiir.txnagent.execution.utils.FileUtil;
import org.dbiir.txnagent.execution.validation.ValidationMetaTable;
import org.dbiir.txnagent.worker.MetaWorker;
import org.dbiir.txnagent.worker.StatisticsWorker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TxnSailsServer {
  private static final Logger logger = LoggerFactory.getLogger(TxnSailsServer.class);
  private static int DEFAULT_AUXILIARY_THREAD_NUM = 16; // used for
  private static Thread flushThread;
  private static ServerSocket serverSocket;
  private static ExecutorService threadPool;
  private static boolean running = true;

  public static void main(String[] args)
      throws InterruptedException, ParseException, SQLException, IOException {
    CommandLineParser parser = new DefaultParser();
    Options options =
        new Options()
            .addOption("c", "config", true, "[required] Workload configuration file")
            .addOption("d", "directory", true, "Base directory for the meta files")
            .addOption("s", "schema", true, "Base directory for the schema sql files")
            .addOption("t", "partition", true, "Partition configuration file")
            .addOption("p", "phase", true, "Online predict or offline training");

    CommandLine argsLine = parser.parse(options, args);

    String schemaFile = argsLine.getOptionValue("s").trim();
    MetaWorker.getINSTANCE().setSchema(new SchemaInfo(schemaFile));

    String configFile = argsLine.getOptionValue("c").trim();
    JacksonXmlConfiguration xmlConfig = buildConfiguration(configFile);

    // get the number of available cores
    int availableProcessors = Runtime.getRuntime().availableProcessors();
    MetaWorker.MAX_AVAILABLE_CORES = (int) Math.ceil(availableProcessors * 0.75);

    try {
      releasePostgreSQLTransactions();
    } catch (Exception e) {
      System.out.println("Release postgres prepared transactions failed: " + e.getMessage());
    }

    AtomicInteger genWorkerId = new AtomicInteger(0);
    WorkloadConfiguration workloadConfiguration = loadConfiguration(xmlConfig);
    List<Connection> auxiliaryConnectionList = makeAuxiliaryConnections(workloadConfiguration);
    ValidationMetaTable.getInstance()
        .initHotspot(workloadConfiguration.getBenchmarkName(), auxiliaryConnectionList);

    String partitionFile = null;
    if (argsLine.hasOption("t")) {
      partitionFile = argsLine.getOptionValue("t").trim();
      PartitionManager.getInstance().init(workloadConfiguration.getBenchmarkName(), partitionFile);
    } else {
      PartitionManager.getInstance().init(workloadConfiguration.getBenchmarkName());
    }

    threadPool =
        new ThreadPoolExecutor(
            512,
            1024,
            60L,
            TimeUnit.SECONDS,
            new LinkedBlockingQueue<>(1000),
            Executors.defaultThreadFactory(),
            new ThreadPoolExecutor.AbortPolicy());

    String statisticsDirectory = "statistics";
    if (argsLine.hasOption("d")) {
      statisticsDirectory = argsLine.getOptionValue("d").trim();
      if (statisticsDirectory.contains("results")) {
        statisticsDirectory = statisticsDirectory.replace("results", "statistics");
        int lastIndex = statisticsDirectory.lastIndexOf("_");

        if (lastIndex != -1) {
          statisticsDirectory = statisticsDirectory.substring(0, lastIndex - 1);
        }
        statisticsDirectory += "/";
      }
    }
    FileUtil.makeDirIfNotExists(statisticsDirectory);
    // init statistics worker with more information
    StatisticsWorker statisticsWorker =
        new StatisticsWorker(statisticsDirectory, workloadConfiguration.getBenchmarkName());
    Thread statisticsWorkerThread = new Thread(statisticsWorker);
    statisticsWorkerThread.start();

    try {
      createFlushThread(
          argsLine,
          workloadConfiguration.getBenchmarkName(),
          workloadConfiguration.getConcurrencyControlType());
      serverSocket = new ServerSocket(9876);
      while (running) {
        try {
          Socket clientSocket = serverSocket.accept();
          clientSocket.setKeepAlive(true);
          threadPool.submit(
              new ClientHandler(clientSocket, workloadConfiguration, genWorkerId.addAndGet(1)));
        } catch (IOException e) {
          if (running) {
            System.out.println(List.of(e.getStackTrace()));
          }
        }
      }
    } finally {
      statisticsWorkerThread.interrupt();
      //      finishFlushThread();
      threadPool.shutdown();
      //      closeServer();
    }
  }

  public static void releasePostgreSQLTransactions() throws ClassNotFoundException, SQLException {
    long startTime = System.nanoTime();
    String[] ips = {"127.0.0.1"};

    Class.forName("org.postgresql.Driver");
    for (String ip : ips) {
      String url = "jdbc:postgresql://" + ip + ":5432/ycsb";
      String username = "zqy";
      String password = "Ss123!@#";

      Connection connection = DriverManager.getConnection(url, username, password);
      Statement statement = connection.createStatement();
      String sql = "SELECT gid FROM pg_prepared_xacts where database = current_database()";
      ResultSet resultSet = statement.executeQuery(sql);
      while (resultSet.next()) {
        String xid = resultSet.getString(1);
        sql = "ROLLBACK PREPARED '" + xid + "'";
        System.out.println(sql);
        Statement stmt = connection.createStatement();
        stmt.execute(sql);
        stmt.close();
      }

      resultSet.close();
      statement.close();
      connection.close();
    }
    long endTime = System.nanoTime();
    System.out.println(
        "Release postgres transaction, time consuming: " + (endTime - startTime) / 1000 + " us");
  }

  public static JacksonXmlConfiguration buildConfiguration(String filename) throws IOException {
    //    String currentPath = System.getProperty("user.dir");
    //    System.out.println("Current working directory: " + currentPath);
    //    System.out.println("Loading configuration from " + filename);

    XmlMapper xmlMapper = new XmlMapper();
    return xmlMapper.readValue(new File(filename), JacksonXmlConfiguration.class);
  }

  private static WorkloadConfiguration loadConfiguration(JacksonXmlConfiguration xmlConfig) {
    // ----------------------------------------------------------------
    // BEGIN LOADING BENCHMARK CONFIGURATION
    // Simplify the evaluation, the config file is provided by application, TxnSails would provide
    // an api in future
    // ----------------------------------------------------------------

    WorkloadConfiguration wrkld = new WorkloadConfiguration();
    wrkld.setXmlConfig(xmlConfig);

    // Pull in database configuration
    wrkld.setDatabaseType(DatabaseType.get(xmlConfig.getType()));
    wrkld.setDriverClass(xmlConfig.getDriver());
    wrkld.setUrl(xmlConfig.getUrl());
    wrkld.setUsername(xmlConfig.getUsername());
    wrkld.setPassword(xmlConfig.getPassword());
    wrkld.setRandomSeed(xmlConfig.getRandomSeed());
    wrkld.setBatchSize(xmlConfig.getBatchsize());
    wrkld.setMaxRetries(xmlConfig.getMaxRetries());
    wrkld.setNewConnectionPerTxn(xmlConfig.isNewConnectionPerTxn());
    wrkld.setReconnectOnConnectionFailure(xmlConfig.isReconnectOnConnectionFailure());
    wrkld.setBenchmarkName(xmlConfig.getBenchmark().toLowerCase());

    int terminals = xmlConfig.getTerminals();
    wrkld.setTerminals(terminals);

    wrkld.setScaleFactor(xmlConfig.getScalefactor());
    String type = xmlConfig.getConcurrencyControlType();
    wrkld.setConcurrencyControlType(type);

    return wrkld;
  }

  private static List<Connection> makeAuxiliaryConnections(WorkloadConfiguration workConf)
      throws SQLException {
    List<Connection> connectionList = new ArrayList<>(16);
    for (int i = 0; i < DEFAULT_AUXILIARY_THREAD_NUM; i++) {
      if (StringUtils.isEmpty(workConf.getUsername())) {
        connectionList.add(DriverManager.getConnection(workConf.getUrl()));
      } else {
        connectionList.add(
            DriverManager.getConnection(
                workConf.getUrl(), workConf.getUsername(), workConf.getPassword()));
      }
    }
    return connectionList;
  }

  private static void createFlushThread(CommandLine argsLine, String benchmark, CCType ccType) {
    String metaDirectory = "metas";
    boolean onlinePredict = false;
    if (argsLine.hasOption("d")) {
      metaDirectory = argsLine.getOptionValue("d").trim();
      if (metaDirectory.contains("results")) {
        metaDirectory = metaDirectory.replace("results", "metas");
        int lastIndex = metaDirectory.lastIndexOf("_");

        if (lastIndex != -1) {
          metaDirectory = metaDirectory.substring(0, lastIndex - 1);
        }
        metaDirectory += "/";
      }
    }

    if (argsLine.hasOption("p")) {
      if (argsLine.getOptionValue("p").contains("online")) {
        onlinePredict = true;
      }
    }

    //    FileUtil.makeDirIfNotExists(metaDirectory);
    //    logger.info("Create Flush Thread");
    //    flushThread = new Thread(new Flusher(benchmark, metaDirectory, ccType, onlinePredict));
    //    flushThread.start();
  }

  //  private static void finishFlushThread() {
  //    flushThread.interrupt();
  //  }

  public static void closeServer() throws IOException {
    running = false;
    if (serverSocket != null && !serverSocket.isClosed()) {
      serverSocket.close();
    }
  }
}
