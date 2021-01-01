package org.apache.spark.mllib.linalg.distributed

import com.holdenkarau.spark.testing.SharedSparkContext
import org.scalatest.FunSuite

class ExtendedBlockMatrixTest extends FunSuite with SharedSparkContext {
	
	def toCoordinateMatrix(data: Seq[MatrixEntry], nbRow: Long, nbCol: Long): CoordinateMatrix = {
		new CoordinateMatrix(sc.parallelize(data), nbRow, nbCol)
	}
	
	def toExtendedBlockMatrix(data: Seq[MatrixEntry], nbRow: Long, nbCol: Long): ExtendedBlockMatrix = {
		toCoordinateMatrix(data, nbRow, nbCol).toBlockMatrix()
	}
	
	val m1Data = Seq(
		MatrixEntry(0, 0, 1.0),
		MatrixEntry(0, 1, 4.0),
		MatrixEntry(1, 0, 3.0),
		MatrixEntry(1, 1, 6.0)
	)
	
	val m2Data = Seq(
		MatrixEntry(0, 0, 2.0),
		MatrixEntry(0, 1, 2.0),
		MatrixEntry(1, 0, 2.0),
		MatrixEntry(1, 1, 2.0)
	)
	
	val m3Data = Seq(
		MatrixEntry(0, 0, 3.0),
		MatrixEntry(0, 1, 3.0),
		MatrixEntry(1, 0, 3.0),
		MatrixEntry(1, 1, 3.0)
	)
	
	test("test Hadamard multiplication") {
		val m1 = toExtendedBlockMatrix(m1Data, 2, 2)
		val m2 = toExtendedBlockMatrix(m2Data, 2, 2)
		val result = m1.hadamard(m2)
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 2.0),
			MatrixEntry(0, 1, 8.0),
			MatrixEntry(1, 0, 6.0),
			MatrixEntry(1, 1, 12.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(result.toSparseBreeze() === compareResult.toSparseBreeze())
	}
	
	test("test norm of matrix") {
		val m1 = toExtendedBlockMatrix(m1Data, 2, 2)
		val result = m1.norm()
		
		assert(result(0) == 4.0)
		assert(result(1) == 10.0)
	}
	
	test("apply operation globally") {
		val m1 = toExtendedBlockMatrix(m1Data, 2, 2)
		val result = m1.applyOperation(m => m * 2.0)
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 2.0),
			MatrixEntry(0, 1, 8.0),
			MatrixEntry(1, 0, 6.0),
			MatrixEntry(1, 1, 12.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(result.toSparseBreeze() === compareResult.toSparseBreeze())
	}
	
	test("test MTTKRP with CoordinateMatrix") {
		val dataMT = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1, 0, 2.0),
			MatrixEntry(0, 1, 3.0),
			MatrixEntry(1, 1, 4.0),
			MatrixEntry(0, 2, 5.0),
			MatrixEntry(1, 2, 6.0),
			MatrixEntry(0, 3, 7.0),
			MatrixEntry(1, 3, 8.0),
			MatrixEntry(0, 7, 9.0),
			MatrixEntry(1, 7, 10.0)
		)
		val mt = toCoordinateMatrix(dataMT, 8, 2).entries.keyBy(entry => math.ceil(entry.i / 1024).toInt)
		val m1 = toExtendedBlockMatrix(m1Data, 2, 2)
		val m2 = toExtendedBlockMatrix(m2Data, 2, 2)
		val m3 = toExtendedBlockMatrix(m3Data, 2, 2)
		val result = ExtendedBlockMatrix.mttkrp(mt, Array(m3, m2, m1), Array(2, 2, 2), 2, 2)
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 258.0),
			MatrixEntry(0, 1, 708.0),
			MatrixEntry(1, 0, 300.0),
			MatrixEntry(1, 1, 840.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(ExtendedBlockMatrix.fromBlockMatrix(result).toSparseBreeze() === compareResult.toSparseBreeze())
	}
	
	test("test MTTKRP with CoordinateMatrix and 2 blocks") {
		val dataM1 = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1025, 1, 10.0)
		)
		val dataM2 = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1, 1, 2.0)
		)
		val dataMT = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1, 0, 2.0),
			MatrixEntry(0, 1, 3.0),
			MatrixEntry(1, 1, 4.0),
			MatrixEntry(0, 2, 5.0),
			MatrixEntry(1, 2, 6.0),
			MatrixEntry(0, 3, 7.0),
			MatrixEntry(1, 3, 8.0),
			MatrixEntry(0, 7, 9.0),
			MatrixEntry(1, 7, 10.0),
			MatrixEntry(0, 2051, 3)
		)
		val mt = toCoordinateMatrix(dataMT, 2, 2052).entries.keyBy(entry => math.ceil(entry.i / 1024).toInt)
		val m1 = toExtendedBlockMatrix(dataM1, 1026, 2)
		val m2 = toExtendedBlockMatrix(dataM2, 2, 2)
		val result = ExtendedBlockMatrix.mttkrp(mt, Array(m2, m1), Array(2, 1026), 2, 2)
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(0, 1, 60.0),
			MatrixEntry(1, 0, 2.0),
			MatrixEntry(1, 1, 0.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(ExtendedBlockMatrix.fromBlockMatrix(result).toSparseBreeze() === compareResult.toSparseBreeze())
	}
}
