/***************************************************************************
 *  Copyright (C) 2013 by H-Store Project                                  *
 *  Brown University                                                       *
 *  Massachusetts Institute of Technology                                  *
 *  Yale University                                                        *
 *                                                                         *
 *  Permission is hereby granted, free of charge, to any person obtaining  *
 *  a copy of this software and associated documentation files (the        *
 *  "Software"), to deal in the Software without restriction, including    *
 *  without limitation the rights to use, copy, modify, merge, publish,    *
 *  distribute, sublicense, and/or sell copies of the Software, and to     *
 *  permit persons to whom the Software is furnished to do so, subject to  *
 *  the following conditions:                                              *
 *                                                                         *
 *  The above copyright notice and this permission notice shall be         *
 *  included in all copies or substantial portions of the Software.        *
 *                                                                         *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        *
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     *
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. *
 *  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR      *
 *  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,  *
 *  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR  *
 *  OTHER DEALINGS IN THE SOFTWARE.                                        *
 ***************************************************************************/

package org.dbiir.txnagent.common.constants;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public abstract class SmallBankConstants {

  // ----------------------------------------------------------------
  // TABLE NAMES
  // ----------------------------------------------------------------
  public static final String TABLENAME_ACCOUNTS = "accounts";
  public static final String TABLENAME_SAVINGS = "savings";
  public static final String TABLENAME_CHECKING = "checking";
  public static final String TABLENAME_CONFLICT = "conflict";
  public static final HashMap<String, Integer> TABLENAME_TO_INDEX = new HashMap<>(4);

  static {
    TABLENAME_TO_INDEX.put(TABLENAME_ACCOUNTS, 0);
    TABLENAME_TO_INDEX.put(TABLENAME_SAVINGS, 1);
    TABLENAME_TO_INDEX.put(TABLENAME_CHECKING, 2);
  }

  public static final HashMap<String, Integer> TABLENAME_TO_HASH_SIZE = new HashMap<>(4);

  static {
    TABLENAME_TO_HASH_SIZE.put(TABLENAME_ACCOUNTS, 1000000);
    TABLENAME_TO_HASH_SIZE.put(TABLENAME_SAVINGS, 1000000);
    TABLENAME_TO_HASH_SIZE.put(TABLENAME_CHECKING, 1000000);
  }

  // ----------------------------------------------------------------
  // ACCOUNT INFORMATION
  // ----------------------------------------------------------------

  // Default number of customers in bank
  public static final int NUM_ACCOUNTS = 1000000;

  public static final boolean HOTSPOT_USE_FIXED_SIZE = true;
  public static final double HOTSPOT_PERCENTAGE = 25; // [0% - 100%]
  public static final int HOTSPOT_FIXED_SIZE = 100; // fixed number of tuples

  // ----------------------------------------------------------------
  // ADDITIONAL CONFIGURATION SETTINGS
  // ----------------------------------------------------------------

  // Initial balance amount
  // We'll just make it really big so that they never run out of money
  public static final int MIN_BALANCE = 10000;
  public static final int MAX_BALANCE = 50000;

  // ----------------------------------------------------------------
  // PROCEDURE PARAMETERS
  // These amounts are from the original code
  // ----------------------------------------------------------------
  public static final double PARAM_SEND_PAYMENT_AMOUNT = 5.0d;
  public static final double PARAM_DEPOSIT_CHECKING_AMOUNT = 1.3d;
  public static final double PARAM_TRANSACT_SAVINGS_AMOUNT = 20.20d;
  public static final double PARAM_WRITE_CHECK_AMOUNT = 5.0d;

  public static final String getSavingVersion =
      "SELECT vid FROM " + TABLENAME_SAVINGS + " WHERE CUSTID = %d";
  public static final String getCheckingVersion =
      "SELECT vid FROM " + TABLENAME_CHECKING + " WHERE CUSTID = %d";

  public static int getHashSize(String tableName) {
    return TABLENAME_TO_HASH_SIZE.get(tableName);
  }

  public static int calculateUniqueId(HashMap<String, Integer> keys, String tableName) {
    int res = -1;
    switch (tableName) {
      case TABLENAME_ACCOUNTS:
        if (keys.containsKey("CUSTID")) {
          res = keys.get("CUSTID");
        } else if (keys.containsKey("NAME")) {
          res = keys.get("NAME");
        }
        break;
      case TABLENAME_CHECKING, TABLENAME_SAVINGS:
        res = keys.get("CUSTID");
        break;
      default:
        System.out.println("Unknown relation name: " + tableName);
        break;
    }
    return res;
  }

  public static String getLatestVersion(String tableName, int validationId) {
    int latestVersion = -1;
    String finalSQL = "";
    switch (tableName) {
      case TABLENAME_ACCOUNTS:
        break;
      case TABLENAME_CHECKING:
        finalSQL = getCheckingVersion.formatted(validationId);
        break;
      case TABLENAME_SAVINGS:
        finalSQL = getSavingVersion.formatted(validationId);
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
    relationToPartitionCount.put(TABLENAME_ACCOUNTS, 4);
    relationToPartitionCount.put(TABLENAME_SAVINGS, 4);
    relationToPartitionCount.put(TABLENAME_CHECKING, 4);
    updatePartitionOffset();
  }

  private static void updatePartitionOffset() {
    relationToPartitionOffset.put(TABLENAME_ACCOUNTS, 0);
    relationToPartitionOffset.put(
        TABLENAME_SAVINGS,
        relationToPartitionOffset.get(TABLENAME_ACCOUNTS)
            + relationToPartitionCount.get(TABLENAME_ACCOUNTS));
    relationToPartitionOffset.put(
        TABLENAME_CHECKING,
        relationToPartitionOffset.get(TABLENAME_SAVINGS)
            + relationToPartitionCount.get(TABLENAME_SAVINGS));
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
