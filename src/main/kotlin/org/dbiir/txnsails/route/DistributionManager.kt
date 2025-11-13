package org.dbiir.txnsails.route

import org.dbiir.txnsails.partition.PartitionInfo

object DistributionManager {
    private val partitions: MutableList<PartitionInfo> = mutableListOf()

    init {
        println("DistributionManager initialized.")
    }

    fun addPartition(partitionInfo: PartitionInfo) {
        partitions.add(partitionInfo)
    }

}