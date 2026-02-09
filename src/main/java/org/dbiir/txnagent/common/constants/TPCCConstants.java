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

package org.dbiir.txnagent.common.constants;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public abstract class TPCCConstants {
  public static final String TABLENAME_DISTRICT = "district";
  public static final String TABLENAME_WAREHOUSE = "warehouse";
  public static final String TABLENAME_ITEM = "item";
  public static final String TABLENAME_STOCK = "stock";
  public static final String TABLENAME_CUSTOMER = "customer";
  public static final String TABLENAME_HISTORY = "history";
  public static final String TABLENAME_OPENORDER = "oorder";
  public static final String TABLENAME_ORDERLINE = "order_line";
  public static final String TABLENAME_NEWORDER = "new_order";

  public static final String getWareHouseVersion =
      "SELECT vid FROM " + TABLENAME_WAREHOUSE + " WHERE W_ID = %d";
  public static final String getDistrictVersion =
      "SELECT vid FROM " + TABLENAME_DISTRICT + " WHERE D_W_ID = %d AND D_ID = %d";
  public static final String getCustomerVersion =
      "SELECT vid FROM " + TABLENAME_CUSTOMER + " WHERE C_W_ID = %d AND C_D_ID = %d AND C_ID = %d";
  public static final String getStockVersion =
      "SELECT vid FROM " + TABLENAME_STOCK + " WHERE S_W_ID = %d AND S_I_ID = %d";
  public static final String getOpenOrderVersion =
      "SELECT vid FROM " + TABLENAME_OPENORDER + " WHERE O_W_ID = %d AND O_D_ID = %d AND O_ID = %d";
  public static final String getOrderLineVersion =
      "SELECT vid FROM "
          + TABLENAME_ORDERLINE
          + " WHERE OL_W_ID = %d AND OL_D_ID = %d AND OL_O_ID = %d";

  public static final int configWhseCount = 32;
  public static final int configItemCount = 100000; // tpc-c std = 100,000
  public static final int configDistPerWhse = 10; // tpc-c std = 10
  public static final int configCustPerDist = 3000; // tpc-c std = 3,000

  /** An invalid item id used to rollback a new order transaction. */
  public static final int INVALID_ITEM_ID = -12345;

  public static final HashMap<String, Integer> TABLENAME_TO_INDEX = new HashMap<>(12);

  static {
    TABLENAME_TO_INDEX.put(TABLENAME_WAREHOUSE, 0);
    TABLENAME_TO_INDEX.put(TABLENAME_DISTRICT, 1);
    TABLENAME_TO_INDEX.put(TABLENAME_ITEM, 2);
    TABLENAME_TO_INDEX.put(TABLENAME_STOCK, 4);
    TABLENAME_TO_INDEX.put(TABLENAME_CUSTOMER, 8);
    TABLENAME_TO_INDEX.put(TABLENAME_OPENORDER, 16);
    TABLENAME_TO_INDEX.put(TABLENAME_ORDERLINE, 32);
    TABLENAME_TO_INDEX.put(TABLENAME_NEWORDER, 64);
  }

  public static final HashMap<String, Integer> TABLENAME_TO_HASH_SIZE = new HashMap<>(8);

  static {
    // TODO: Review this number
    TABLENAME_TO_HASH_SIZE.put(TABLENAME_WAREHOUSE, configWhseCount);
    TABLENAME_TO_HASH_SIZE.put(TABLENAME_DISTRICT, configWhseCount * configDistPerWhse);
    TABLENAME_TO_HASH_SIZE.put(
        TABLENAME_CUSTOMER, configWhseCount * configDistPerWhse * configCustPerDist);
    TABLENAME_TO_HASH_SIZE.put(TABLENAME_STOCK, configWhseCount * configItemCount);
    TABLENAME_TO_HASH_SIZE.put(
        TABLENAME_OPENORDER, configWhseCount * configDistPerWhse * configCustPerDist);
    TABLENAME_TO_HASH_SIZE.put(
        TABLENAME_ORDERLINE, configWhseCount * configDistPerWhse * configCustPerDist);
  }

  public static int getHashSize(String tableName) {
    return TABLENAME_TO_HASH_SIZE.get(tableName);
  }

  public static int calculateUniqueId(HashMap<String, Integer> keys, String tableName) {
    int res = -1;
    switch (tableName) {
      case TABLENAME_WAREHOUSE:
        res = keys.get("W_ID") - 1;
        break;
      case TABLENAME_DISTRICT:
        res = (keys.get("D_W_ID") - 1) * configDistPerWhse + keys.get("D_ID") - 1;
        break;
      case TABLENAME_CUSTOMER:
        res =
            (keys.get("C_W_ID") - 1) * configDistPerWhse * configCustPerDist
                + (keys.get("C_D_ID") - 1) * configCustPerDist
                + keys.get("C_ID")
                - 1;
        break;
      case TABLENAME_STOCK:
        res = (keys.get("S_W_ID") - 1) * configItemCount + keys.get("S_I_ID") - 1;
        break;
      case TABLENAME_OPENORDER:
        res =
            (keys.get("O_W_ID") - 1) * configDistPerWhse * configCustPerDist
                + (keys.get("O_D_ID") - 1) * configCustPerDist
                + keys.get("O_ID")
                - 1;
        break;
      case TABLENAME_ORDERLINE:
        res =
            (keys.get("OL_W_ID") - 1) * configDistPerWhse * configCustPerDist
                + (keys.get("OL_D_ID") - 1) * configCustPerDist
                + keys.get("OL_O_ID")
                - 1;
        break;
      default:
        System.out.println("Unknown relation name: " + tableName);
        break;
    }
    return res;
  }

  // return the index of the table
  public static String getLatestVersion(String tableName, int validationId) {
    int latestVersion = -1;
    String finalSQL = "";
    switch (tableName) {
      case TABLENAME_WAREHOUSE:
        finalSQL = getWareHouseVersion.formatted(validationId + 1);
        break;
      case TABLENAME_DISTRICT:
        int d_w_id = (validationId / configDistPerWhse) + 1;
        int d_id = (validationId % configDistPerWhse) + 1;
        finalSQL = getDistrictVersion.formatted(d_w_id, d_id);
        break;
      case TABLENAME_CUSTOMER:
        int c_id = (validationId % configCustPerDist) + 1;
        int c_d_id = ((validationId / configCustPerDist) % configDistPerWhse) + 1;
        int c_w_id = (validationId / (configDistPerWhse * configCustPerDist)) + 1;
        finalSQL = getCustomerVersion.formatted(c_w_id, c_d_id, c_id);
        break;
      case TABLENAME_STOCK:
        int s_w_id = (validationId / configItemCount) + 1;
        int s_i_id = (validationId % configItemCount) + 1;
        finalSQL = getStockVersion.formatted(s_w_id, s_i_id);
        break;
      case TABLENAME_OPENORDER:
        int o_id = (validationId % configCustPerDist) + 1;
        int o_d_id = ((validationId / configCustPerDist) % configDistPerWhse) + 1;
        int o_w_id = (validationId / (configDistPerWhse * configCustPerDist)) + 1;
        finalSQL = getOpenOrderVersion.formatted(o_w_id, o_d_id, o_id);
        break;
      case TABLENAME_ORDERLINE:
        int ol_o_id = (validationId % configCustPerDist) + 1;
        int ol_d_id = ((validationId / configCustPerDist) % configDistPerWhse) + 1;
        int ol_w_id = (validationId / (configDistPerWhse * configCustPerDist)) + 1;
        finalSQL = getOrderLineVersion.formatted(ol_w_id, ol_d_id, ol_o_id);
        break;
      default:
        System.out.println("Unknown relation name: " + tableName);
        break;
    }

    return finalSQL;
  }

  private static final HashMap<String, Integer> relationToPartitionCount = new HashMap<>();
  private static final HashMap<String, Integer> relationToPartitionOffset = new HashMap<>();
  private static final HashMap<String, Integer> relationToPartitionSize = new HashMap<>();

  static {
    relationToPartitionCount.put(TABLENAME_WAREHOUSE, 1);
    relationToPartitionCount.put(TABLENAME_DISTRICT, 1);
    relationToPartitionCount.put(TABLENAME_CUSTOMER, 4);
    relationToPartitionCount.put(TABLENAME_STOCK, 4);
    relationToPartitionCount.put(TABLENAME_OPENORDER, 4);
    relationToPartitionCount.put(TABLENAME_ORDERLINE, 4);
    updatePartitionOffset();
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

  private static void updatePartitionOffset() {
    relationToPartitionOffset.clear();
    relationToPartitionOffset.put(TABLENAME_WAREHOUSE, 0);
    relationToPartitionOffset.put(
        TABLENAME_DISTRICT,
        relationToPartitionOffset.get(TABLENAME_WAREHOUSE)
            + relationToPartitionCount.get(TABLENAME_WAREHOUSE));
    relationToPartitionOffset.put(
        TABLENAME_CUSTOMER,
        relationToPartitionOffset.get(TABLENAME_DISTRICT)
            + relationToPartitionCount.get(TABLENAME_DISTRICT));
    relationToPartitionOffset.put(
        TABLENAME_STOCK,
        relationToPartitionOffset.get(TABLENAME_CUSTOMER)
            + relationToPartitionCount.get(TABLENAME_CUSTOMER));
    relationToPartitionOffset.put(
        TABLENAME_OPENORDER,
        relationToPartitionOffset.get(TABLENAME_STOCK)
            + relationToPartitionCount.get(TABLENAME_STOCK));
    relationToPartitionOffset.put(
        TABLENAME_ORDERLINE,
        relationToPartitionOffset.get(TABLENAME_OPENORDER)
            + relationToPartitionCount.get(TABLENAME_OPENORDER));
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
