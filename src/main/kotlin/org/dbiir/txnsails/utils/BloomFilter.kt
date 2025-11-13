package org.dbiir.txnsails.utils

import java.util.BitSet
import kotlin.math.*

class BloomFilter(
    private val expectedInsertions: Int,
    private val falsePositiveRate: Double = 0.01
) {
    private val bitSet: BitSet
    private val numBits: Int
    private val numHashFunctions: Int

    init {
        require(expectedInsertions > 0) { "Expected insertions must be positive" }
        require(falsePositiveRate > 0.0 && falsePositiveRate < 1.0) {
            "False positive rate must be between 0 and 1"
        }

        this.numBits = calculateOptimalNumBits(expectedInsertions, falsePositiveRate)
        this.numHashFunctions = calculateOptimalNumHashFunctions(expectedInsertions, numBits)
        this.bitSet = BitSet(numBits)

        println("Long Bloom Filter initialized:")
        println("  - Expected insertions: $expectedInsertions")
        println("  - False positive rate: $falsePositiveRate")
        println("  - Bit array size: $numBits bits")
        println("  - Hash functions: $numHashFunctions")
    }

    fun add(number: Long) {
        val hashes = getHashes(number)
        hashes.forEach { hash ->
            bitSet.set(hash % numBits)
        }
    }

    fun addAll(numbers: LongArray) {
        numbers.forEach { add(it) }
    }

    fun addAll(numbers: Collection<Long>) {
        numbers.forEach { add(it) }
    }

    fun mightContain(number: Long): Boolean {
        val hashes = getHashes(number)
        return hashes.all { hash ->
            bitSet.get(hash % numBits)
        }
    }

    fun clear() {
        bitSet.clear()
    }

    fun getStats(): BloomFilterStats {
        val bitsSet = bitSet.cardinality()
        val actualFalsePositiveRate = estimatedFalsePositiveRate()

        return BloomFilterStats(
            expectedInsertions = expectedInsertions,
            actualInsertions = -1,
            numBits = numBits,
            numHashFunctions = numHashFunctions,
            bitsSet = bitsSet,
            fillRatio = bitsSet.toDouble() / numBits,
            estimatedFalsePositiveRate = actualFalsePositiveRate
        )
    }

    fun estimatedFalsePositiveRate(): Double {
        val bitsSet = bitSet.cardinality().toDouble()
        return (1 - exp(-numHashFunctions * bitsSet / numBits)).pow(numHashFunctions.toDouble())
    }

    private fun calculateOptimalNumBits(n: Int, p: Double): Int {
        return max(1, (-n * ln(p) / (ln(2.0) * ln(2.0))).toInt())
    }

    private fun calculateOptimalNumHashFunctions(n: Int, m: Int): Int {
        return max(1, ((m.toDouble() / n) * ln(2.0)).roundToInt())
    }

    private fun getHashes(number: Long): IntArray {
        val hashes = IntArray(numHashFunctions)

        val hash1 = number.hashCode() and 0x7FFFFFFF
        val hash2 = murmurHash(number).toInt() and 0x7FFFFFFF

        for (i in 0 until numHashFunctions) {
            hashes[i] = (hash1 + i * hash2) and 0x7FFFFFFF
        }

        return hashes
    }

    private fun murmurHash(key: Long): Long {
        var h = key
        h = h xor (h ushr 33)
        h *= -0xae502812aa7333L
        h = h xor (h ushr 33)
        h *= -0x3b314601e57a13adL
        h = h xor (h ushr 33)
        return h
    }

    /**
     * 统计信息类
     */
    data class BloomFilterStats(
        val expectedInsertions: Int,
        val actualInsertions: Int,
        val numBits: Int,
        val numHashFunctions: Int,
        val bitsSet: Int,
        val fillRatio: Double,
        val estimatedFalsePositiveRate: Double
    ) {
        override fun toString(): String {
            return """
                Long Bloom Filter Statistics:
                - Expected insertions: $expectedInsertions
                - Actual insertions: ${if (actualInsertions >= 0) actualInsertions else "Unknown"}
                - Bit array size: $numBits bits
                - Hash functions: $numHashFunctions
                - Bits set: $bitsSet
                - Fill ratio: ${"%.2f".format(fillRatio * 100)}%
                - Estimated false positive rate: ${"%.4f".format(estimatedFalsePositiveRate * 100)}%
            """.trimIndent()
        }
    }
}
