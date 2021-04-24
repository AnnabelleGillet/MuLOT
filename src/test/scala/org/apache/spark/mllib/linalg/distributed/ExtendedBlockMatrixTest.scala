package org.apache.spark.mllib.linalg.distributed

import mulot.Tensor
import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite
import org.scalatest.Matchers.convertNumericToPlusOrMinusWrapper

class ExtendedBlockMatrixTest extends FunSuite {
	
	implicit val spark = SparkSession.builder().master("local[*]").getOrCreate()
	spark.sparkContext.setLogLevel("WARN")
	
	def toCoordinateMatrix(data: Seq[MatrixEntry], nbRow: Long, nbCol: Long): CoordinateMatrix = {
		new CoordinateMatrix(spark.sparkContext.parallelize(data), nbRow, nbCol)
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
		val result = m1.normL1()
		
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
		val mt = toCoordinateMatrix(dataMT,2, 8).entries.keyBy(entry => math.ceil(entry.i / 1024).toInt)
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
	
	test("test MTTKRP with DataFrame") {
		import spark.implicits._
		val dataMT = Seq[(Long, Long, Long, Long, Double)](
			(0L, 0L, 0L, 0L, 1.0),
			(1L, 0L, 0L, 0L, 2.0),
			(0L, 0L, 0L, 1L, 3.0),
			(1L, 0L, 0L, 1L, 4.0),
			(0L, 0L, 1L, 0L, 5.0),
			(1L, 0L, 1L, 0L, 6.0),
			(0L, 0L, 1L, 1L, 7.0),
			(1L, 0L, 1L, 1L, 8.0),
			(0L, 1L, 1L, 1L, 9.0),
			(1L, 1L, 1L, 1L, 10.0)
		)
		val mt = dataMT.toDF("row_0", "row_1", "row_2", "row_3", "val")
		val m1 = toExtendedBlockMatrix(m1Data, 2, 2)
		val m2 = toExtendedBlockMatrix(m2Data, 2, 2)
		val m3 = toExtendedBlockMatrix(m3Data, 2, 2)
		val result = ExtendedBlockMatrix.mttkrpDataFrame(mt, Array(m1, m2, m3), Array(2, 2, 2), 0, 2, 2, "val")
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 258.0),
			MatrixEntry(0, 1, 708.0),
			MatrixEntry(1, 0, 300.0),
			MatrixEntry(1, 1, 840.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(ExtendedBlockMatrix.fromBlockMatrix(result).toSparseBreeze() === compareResult.toSparseBreeze())
	}
	
	test("test MTTKRP with CoordinateMatrix and DataFrame") {
		import spark.implicits._
		val dataMT = Seq[(Long, Long, Long, Long, Double)](
			(0L, 0L, 0L, 0L, 1.0),
			(1L, 0L, 0L, 0L, 2.0),
			(0L, 0L, 0L, 1L, 3.0),
			(1L, 0L, 0L, 1L, 4.0),
			(0L, 0L, 1L, 0L, 5.0),
			(1L, 0L, 1L, 0L, 6.0),
			(0L, 0L, 1L, 1L, 7.0),
			(1L, 0L, 1L, 1L, 8.0),
			(0L, 1L, 1L, 1L, 9.0),
			(1L, 1L, 1L, 1L, 10.0)
		)
		val mt = dataMT.toDF("row_0", "row_1", "row_2", "row_3", "val")
		val tensor = Tensor.fromIndexedDataFrame(mt, List(2, 2, 2, 2))
		val m1 = toExtendedBlockMatrix(m1Data, 2, 2)
		val m2 = toExtendedBlockMatrix(m2Data, 2, 2)
		val m3 = toExtendedBlockMatrix(m3Data, 2, 2)
		val result = ExtendedBlockMatrix.mttkrpDataFrame(mt, Array(m1, m2, m3), Array(2, 2, 2), 0, 2, 2, "val")
		val result2 = ExtendedBlockMatrix.mttkrp(tensor.matricization()(0), Array(m1, m2, m3), Array(2, 2, 2), 2, 2)
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 258.0),
			MatrixEntry(0, 1, 708.0),
			MatrixEntry(1, 0, 300.0),
			MatrixEntry(1, 1, 840.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(ExtendedBlockMatrix.fromBlockMatrix(result).toSparseBreeze() === compareResult.toSparseBreeze())
		assert(ExtendedBlockMatrix.fromBlockMatrix(result2).toSparseBreeze() === compareResult.toSparseBreeze())
	}
	
	test("test MTTKRP with DataFrame and 2 blocks") {
		import spark.implicits._
		val dataM1 = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1025, 1, 10.0)
		)
		val dataM2 = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1, 1, 2.0)
		)
		val dataMT = Seq[(Long, Long, Long, Double)](
			(0L, 0L, 0L, 1.0),
			(1L, 0L, 0L, 2.0),
			(0L, 0L, 1L, 3.0),
			(1L, 0L, 1L, 4.0),
			(0L, 1L, 0L, 5.0),
			(1L, 1L, 0L, 6.0),
			(0L, 1L, 1L, 7.0),
			(1L, 1L, 1L, 8.0),
			(0L, 3L, 1L, 9.0),
			(1L, 3L, 1L, 10.0),
			(0L, 1025L, 1L, 3.0)
		)
		val mt = dataMT.toDF("row_0", "row_1", "row_2", "val")//val mt = toCoordinateMatrix(dataMT, 2, 2052).entries.keyBy(entry => math.ceil(entry.i / 1024).toInt)
		val m1 = toExtendedBlockMatrix(dataM1, 1026, 2)
		val m2 = toExtendedBlockMatrix(dataM2, 2, 2)
		val result = ExtendedBlockMatrix.mttkrpDataFrame(mt, Array(m1, m2), Array(1026, 2), 0, 2, 2, "val")
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(0, 1, 60.0),
			MatrixEntry(1, 0, 2.0),
			MatrixEntry(1, 1, 0.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(ExtendedBlockMatrix.fromBlockMatrix(result).toSparseBreeze() === compareResult.toSparseBreeze())
	}
	
	test("test MTTKRP with CoordinateMatrix from DataFrame and 2 blocks") {
		import spark.implicits._
		val dataM1 = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1025, 1, 10.0)
		)
		val dataM2 = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(1, 1, 2.0)
		)
		val dataMT = Seq[(Long, Long, Long, Double)](
			(0L, 0L, 0L, 1.0),
			(1L, 0L, 0L, 2.0),
			(0L, 0L, 1L, 3.0),
			(1L, 0L, 1L, 4.0),
			(0L, 1L, 0L, 5.0),
			(1L, 1L, 0L, 6.0),
			(0L, 1L, 1L, 7.0),
			(1L, 1L, 1L, 8.0),
			(0L, 3L, 1L, 9.0),
			(1L, 3L, 1L, 10.0),
			(0L, 1025L, 1L, 3.0)
		)
		val mt = dataMT.toDF("row_0", "row_1", "row_2", "val")//val mt = toCoordinateMatrix(dataMT, 2, 2052).entries.keyBy(entry => math.ceil(entry.i / 1024).toInt)
		val tensor = Tensor.fromIndexedDataFrame(mt, List(2, 1026, 2))
		val m1 = toExtendedBlockMatrix(dataM1, 1026, 2)
		val m2 = toExtendedBlockMatrix(dataM2, 2, 2)
		val result = ExtendedBlockMatrix.mttkrpDataFrame(mt, Array(m1, m2), Array(1026, 2), 0, 2, 2, "val")
		val result2 = ExtendedBlockMatrix.mttkrp(tensor.matricization()(0), Array(m1, m2), Array(1026, 2), 2, 2)
		
		val compareResultData = Seq(
			MatrixEntry(0, 0, 1.0),
			MatrixEntry(0, 1, 60.0),
			MatrixEntry(1, 0, 2.0),
			MatrixEntry(1, 1, 0.0)
		)
		val compareResult = toExtendedBlockMatrix(compareResultData, 2, 2)
		
		assert(ExtendedBlockMatrix.fromBlockMatrix(result).toSparseBreeze() === compareResult.toSparseBreeze())
		assert(ExtendedBlockMatrix.fromBlockMatrix(result2).toSparseBreeze() === compareResult.toSparseBreeze())
	}
	
	test("test pinv of Breeze and Spark") {
		val m = ExtendedBlockMatrix.gaussian(5, 5)
		
		val breezePinv = m.pinverse().toSparseBreeze()
		val sparkPinv = m.sparkPinverse().toSparseBreeze()
		
		for (i <- 0 until 5; j <- 0 until 5) {
			assert(breezePinv(i, j) === sparkPinv(i, j) +- 1E-4)
		}
	}
}
