/*
 * Copyright 2020 by OLTPBenchmark Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.dbiir.txnsails.common.constants;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public abstract class YCSBConstants {

  public static final int RECORD_COUNT = 1000;

  public static final int NUM_FIELDS = 10;

  /**
   * The max size of each field in the USERTABLE. NOTE: If you increase this value here in the code,
   * then you must update all the DDL files.
   */
  public static final int MAX_FIELD_SIZE = 100; // chars

  /** How many records will each thread load. */
  public static final int THREAD_BATCH_SIZE = 50000;

  public static final int MAX_SCAN = 1000;

  public static final String TABLE_NAME = "usertable";

  public static final HashMap<String, Integer> TABLENAME_TO_INDEX = new HashMap<>(1);

  public static String getUserTableVersion =
      "SELECT vid FROM " + TABLE_NAME + " WHERE YCSB_KEY = %d";

  static {
    TABLENAME_TO_INDEX.put(TABLE_NAME, 1);
  }

  public static final HashMap<String, Integer> TABLENAME_TO_HASH_SIZE = new HashMap<>(1);

  static {
    TABLENAME_TO_HASH_SIZE.put(TABLE_NAME, 1000000);
  }

  public static int getHashSize(String tableName) {
    return TABLENAME_TO_HASH_SIZE.get(tableName);
  }

  public static int calculateUniqueId(HashMap<String, Integer> keys, String tableName) {
    int res = -1;
    if (tableName.equals(TABLE_NAME)) {
      res = keys.get("YCSB_KEY");
    } else {
      System.out.println("Unknown relation name: " + tableName);
    }
    return res;
  }

  public static String getLatestVersion(String tableName, int validationId) {
    String finalSQL = "";
    if (tableName.equals(TABLE_NAME)) {
      finalSQL = getUserTableVersion.formatted(validationId);
    } else {
      System.out.println("Unknown relation name: " + tableName);
    }

    return finalSQL;
  }

  private static final HashMap<String, Integer> relationToPartitionCount = new HashMap<>();
  private static final HashMap<String, Integer> relationToPartitionOffset = new HashMap<>();
  private static final HashMap<String, Integer> relationToPartitionSize = new HashMap<>();

  static {
    relationToPartitionCount.put(TABLE_NAME, 4);
    updatePartitionOffset();
  }

  private static void updatePartitionOffset() {
    relationToPartitionOffset.put(TABLE_NAME, 0);
  }

  public static void loadPartitioningInfo(
      HashMap<String, Integer> tablePartitionCountMap,
      HashMap<String, Integer> tablePartitionSizeMap) {
    for (String tableName : tablePartitionCountMap.keySet()) {
      if (!relationToPartitionCount.containsKey(tableName)) {
        System.out.println("Unknown table name when loading partitioning info: " + tableName);
        continue;
      }
      relationToPartitionCount.put(tableName, tablePartitionCountMap.get(tableName));
      relationToPartitionSize.put(tableName, tablePartitionSizeMap.get(tableName));
    }
    updatePartitionOffset();
  }

  public static int transferToGlobalPartitionId(String tableName, int partitionId) {
    if (!relationToPartitionOffset.containsKey(tableName)) {
      System.out.println("Unknown table name when getting global partition id: " + tableName);
      return -1;
    }
    int offset = relationToPartitionOffset.get(tableName);
    return partitionId + offset;
  }

  public static int transferToOffsetPartitionId(String tableName, int globalPartitionId) {
    if (!relationToPartitionOffset.containsKey(tableName)) {
      System.out.println("Unknown table name when getting offset partition id: " + tableName);
      return -1;
    }
    int offset = relationToPartitionOffset.get(tableName);
    return globalPartitionId - offset;
  }

  public static int getGlobalPartitionIdByKey(String tableName, int key) {
    if (!relationToPartitionOffset.containsKey(tableName)) {
      System.out.println("Unknown table name when getting offset partition id: " + tableName);
      return -1;
    }
    int offset = relationToPartitionOffset.get(tableName);
    int partitionId = key / relationToPartitionSize.get(tableName);
    return partitionId + offset;
  }

  public static List<Integer> getGlobalPartitionIds() {
    List<Integer> globalPartitionIds = new ArrayList<>();
    for (String tableName : relationToPartitionCount.keySet()) {
      int partitionCount = relationToPartitionCount.get(tableName);
      int offset = relationToPartitionOffset.get(tableName);
      for (int i = 0; i < partitionCount; i++) {
        globalPartitionIds.add(offset + i);
      }
    }
    return globalPartitionIds;
  }
}
