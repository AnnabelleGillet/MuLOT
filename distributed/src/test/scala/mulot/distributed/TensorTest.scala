package mulot.distributed

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.must.Matchers.contain
import org.scalatest.matchers.should.Matchers.{convertToAnyShouldWrapper, equal}

class TensorTest extends AnyFunSuite {
	implicit val spark = SparkSession.builder().master("local[*]").getOrCreate()
	spark.sparkContext.setLogLevel("WARN")
	
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
		(1L, 1L, 1L, 1L, 10.0),
		(2L, 1L, 1L, 1L, 11.0)
	)
	val mt = dataMT.toDF("row_0", "row_1", "row_2", "row_3", "val")
	val tensor = Tensor.fromIndexedDataFrame(mt, Array(3L, 2L, 2L, 2L))
	
	test("test reindexation of tensor") {
		val dataTensor1 = Seq[(Int, Int, Double)](
			(0, 0, 1.0),
			(2, 2, 9.0),
			(1, 0, 2.0),
			(2, 0, 3.0),
			(0, 1, 4.0),
			(1, 1, 5.0),
			(2, 1, 6.0),
			(0, 2, 7.0),
			(1, 2, 8.0)
		)
		val tensor1 = Tensor(dataTensor1.toDF("Dim1", "Dim2", "val"))
		
		val dataTensor2 = Seq[(Int, Int, Double)](
			(0, 0, 1.0),
			(1, 0, 2.0),
			(2, 0, 3.0),
			(0, 1, 4.0),
			(1, 1, 5.0),
			(2, 1, 6.0),
			(0, 2, 7.0),
			(1, 2, 8.0),
			(2, 2, 9.0)
		)
		val tensor2 = Tensor.fromIndexedDataFrame(dataTensor2.toDF("row_0", "row_1", "val"), Array(3, 3))
		
		for (i <- 0 until 2; j <- 0 until 2) {
			val newIndex = Tensor.reindexDimension(Array((tensor1, i), (tensor2, j)))
			val newTensor1 = tensor1.reindex(i, newIndex)
			val newTensor2 = tensor2.reindex(j, newIndex)
			newTensor1.dimensionsIndex(i).collect should contain theSameElementsAs newTensor2.dimensionsIndex(j).collect
			var rebuildedTensor1 = newTensor1.data
			for (dim <- 0 until 2) {
				rebuildedTensor1 = rebuildedTensor1.join(newTensor1.dimensionsIndex(dim), col(s"row_$dim") === col("dimIndex"))
					.withColumnRenamed("dimValue", s"Dim${dim + 1}")
					.drop("dimIndex", s"row_$dim")
			}
			rebuildedTensor1 = rebuildedTensor1.select("Dim1", "Dim2", "val")
			rebuildedTensor1.collect() should contain theSameElementsAs dataTensor1.toDF("Dim1", "Dim2", "val").collect()
			var rebuildedTensor2 = newTensor2.data
			for (dim <- 0 until 2) {
				rebuildedTensor2 = rebuildedTensor2.join(newTensor2.dimensionsIndex(dim), col(s"row_$dim") === col("dimIndex"))
					.withColumnRenamed("dimValue", s"Dim${dim + 1}")
					.drop("dimIndex", s"row_$dim")
			}
			rebuildedTensor2 = rebuildedTensor2.select("Dim1", "Dim2", "val")
			rebuildedTensor2.collect() should contain theSameElementsAs dataTensor2.toDF("Dim1", "Dim2", "val").collect()
		}
	}
	
	test("test reindexation of tensor with different sizes of dimensions") {
		val dataTensor1 = Seq[(Int, Int, Double)](
			(0, 0, 1.0),
			(1, 0, 2.0),
			(0, 1, 4.0),
			(1, 1, 5.0)
		)
		val tensor1 = Tensor(dataTensor1.toDF("Dim1", "Dim2", "val"))
		
		val dataTensor2 = Seq[(Int, Int, Double)](
			(0, 0, 1.0),
			(1, 0, 2.0),
			(2, 0, 3.0),
			(0, 1, 4.0),
			(1, 1, 5.0),
			(2, 1, 6.0),
			(0, 2, 7.0),
			(1, 2, 8.0),
			(2, 2, 9.0)
		)
		val tensor2 = Tensor.fromIndexedDataFrame(dataTensor2.toDF("row_0", "row_1", "val"), Array(3, 3))
		
		for (i <- 0 until 2; j <- 0 until 2) {
			val newIndex = Tensor.reindexDimension(Array((tensor1, i), (tensor2, j)))
			val newTensor1 = tensor1.reindex(i, newIndex)
			val newTensor2 = tensor2.reindex(j, newIndex)
			for (k <- 0 until 2) {
				if (k != i) {
					newTensor1.dimensionsSize(k) should equal(2)
				}
				if (k != j) {
					newTensor2.dimensionsSize(k) should equal(3)
				}
			}
			newTensor1.dimensionsSize(i) should equal(3)
			newTensor2.dimensionsSize(j) should equal(3)
			newTensor1.dimensionsIndex(i).collect should contain theSameElementsAs newTensor2.dimensionsIndex(j).collect
			var rebuildedTensor1 = newTensor1.data
			for (dim <- 0 until 2) {
				rebuildedTensor1 = rebuildedTensor1.join(newTensor1.dimensionsIndex(dim), col(s"row_$dim") === col("dimIndex"))
					.withColumnRenamed("dimValue", s"Dim${dim + 1}")
					.drop("dimIndex", s"row_$dim")
			}
			rebuildedTensor1 = rebuildedTensor1.select("Dim1", "Dim2", "val")
			rebuildedTensor1.collect() should contain theSameElementsAs dataTensor1.toDF("Dim1", "Dim2", "val").collect()
			var rebuildedTensor2 = newTensor2.data
			for (dim <- 0 until 2) {
				rebuildedTensor2 = rebuildedTensor2.join(newTensor2.dimensionsIndex(dim), col(s"row_$dim") === col("dimIndex"))
					.withColumnRenamed("dimValue", s"Dim${dim + 1}")
					.drop("dimIndex", s"row_$dim")
			}
			rebuildedTensor2 = rebuildedTensor2.select("Dim1", "Dim2", "val")
			rebuildedTensor2.collect() should contain theSameElementsAs dataTensor2.toDF("Dim1", "Dim2", "val").collect()
		}
	}
	
	test("test matricization mode 0") {
		val dataCompareResult = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(1L, 0L, 2.0),
			(0L, 4L, 3.0),
			(1L, 4L, 4.0),
			(0L, 2L, 5.0),
			(1L, 2L, 6.0),
			(0L, 6L, 7.0),
			(1L, 6L, 8.0),
			(0L, 7L, 9.0),
			(1L, 7L, 10.0),
			(2L, 7L, 11.0)
		)
		val _result = tensor.matricization(0)
		val result = _result.toCoordinateMatrix().entries.toDF()
		assert(_result.numCols() == 8)
		assert(_result.numRows() == 3)
		val resultSeq = result.collect().toSeq
		assert(result.distinct().collect().size == resultSeq.size)
		assert(resultSeq.size == dataCompareResult.size)
		for (row <- result.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
	}
	
	test("test matricization mode 1") {
		val dataCompareResult = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(0L, 1L, 2.0),
			(0L, 6L, 3.0),
			(0L, 7L, 4.0),
			(0L, 3L, 5.0),
			(0L, 4L, 6.0),
			(0L, 9L, 7.0),
			(0L, 10L, 8.0),
			(1L, 9L, 9.0),
			(1L, 10L, 10.0),
			(1L, 11L, 11.0)
		)
		val _result = tensor.matricization(1)
		val result = _result.toCoordinateMatrix().entries.toDF()
		assert(_result.numCols() == 12)
		assert(_result.numRows() == 2)
		val resultSeq = result.collect().toSeq
		assert(result.distinct().collect().size == resultSeq.size)
		assert(resultSeq.size == dataCompareResult.size)
		for (row <- result.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
	}
	
	test("test matricization mode 2") {
		val dataCompareResult = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(0L, 1L, 2.0),
			(0L, 6L, 3.0),
			(0L, 7L, 4.0),
			(1L, 0L, 5.0),
			(1L, 1L, 6.0),
			(1L, 6L, 7.0),
			(1L, 7L, 8.0),
			(1L, 9L, 9.0),
			(1L, 10L, 10.0),
			(1L, 11L, 11.0)
		)
		val _result = tensor.matricization(2)
		val result = _result.toCoordinateMatrix().entries.toDF()
		assert(_result.numCols() == 12)
		assert(_result.numRows() == 2)
		val resultSeq = result.collect().toSeq
		assert(result.distinct().collect().size == resultSeq.size)
		assert(resultSeq.size == dataCompareResult.size)
		for (row <- result.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
	}
	
	test("test chained matricization mode 0") {
		val dataCompareResult = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(1L, 0L, 2.0),
			(0L, 4L, 3.0),
			(1L, 4L, 4.0),
			(0L, 2L, 5.0),
			(1L, 2L, 6.0),
			(0L, 6L, 7.0),
			(1L, 6L, 8.0),
			(0L, 7L, 9.0),
			(1L, 7L, 10.0),
			(2L, 7L, 11.0)
		)
		val _result = tensor.matricization(0)
		val result = _result.toCoordinateMatrix().entries.toDF()
		assert(_result.numCols() == 8)
		assert(_result.numRows() == 3)
		val resultSeq = result.collect().toSeq
		assert(result.distinct().collect().size == resultSeq.size)
		assert(resultSeq.size == dataCompareResult.size)
		for (row <- result.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
		
		val dataCompareResult2 = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(0L, 1L, 2.0),
			(0L, 6L, 3.0),
			(0L, 7L, 4.0),
			(0L, 3L, 5.0),
			(0L, 4L, 6.0),
			(0L, 9L, 7.0),
			(0L, 10L, 8.0),
			(1L, 9L, 9.0),
			(1L, 10L, 10.0),
			(1L, 11L, 11.0)
		)
		val _result2 = tensor.matricization(1)
		val result2 = _result2.toCoordinateMatrix().entries.toDF()
		assert(_result2.numCols() == 12)
		assert(_result2.numRows() == 2)
		val resultSeq2 = result2.collect().toSeq
		assert(result2.distinct().collect().size == resultSeq2.size)
		assert(resultSeq2.size == dataCompareResult2.size)
		for (row <- result2.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult2.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
		
		val dataCompareResult3 = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(0L, 1L, 2.0),
			(0L, 6L, 3.0),
			(0L, 7L, 4.0),
			(1L, 0L, 5.0),
			(1L, 1L, 6.0),
			(1L, 6L, 7.0),
			(1L, 7L, 8.0),
			(1L, 9L, 9.0),
			(1L, 10L, 10.0),
			(1L, 11L, 11.0)
		)
		val _result3 = tensor.matricization(2)
		val result3 = _result3.toCoordinateMatrix().entries.toDF()
		assert(_result3.numCols() == 12)
		assert(_result3.numRows() == 2)
		val resultSeq3 = result3.collect().toSeq
		assert(result3.distinct().collect().size == resultSeq3.size)
		assert(resultSeq3.size == dataCompareResult3.size)
		for (row <- result3.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult3.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
	}
	
	test("test transposed matricization mode 0") {
		val dataCompareResult = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(0L, 1L, 2.0),
			(4L, 0L, 3.0),
			(4L, 1L, 4.0),
			(2L, 0L, 5.0),
			(2L, 1L, 6.0),
			(6L, 0L, 7.0),
			(6L, 1L, 8.0),
			(7L, 0L, 9.0),
			(7L, 1L, 10.0),
			(7L, 2L, 11.0)
		)
		val _result = tensor.matricization(0, true)
		val result = _result.toCoordinateMatrix().entries.toDF()
		assert(_result.numCols() == 3)
		assert(_result.numRows() == 8)
		val resultSeq = result.collect().toSeq
		assert(result.distinct().collect().size == resultSeq.size)
		assert(resultSeq.size == dataCompareResult.size)
		for (row <- result.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
	}
	
	test("test transposed matricization mode 1") {
		val dataCompareResult = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(1L, 0L, 2.0),
			(6L, 0L, 3.0),
			(7L, 0L, 4.0),
			(3L, 0L, 5.0),
			(4L, 0L, 6.0),
			(9L, 0L, 7.0),
			(10L, 0L, 8.0),
			(9L, 1L, 9.0),
			(10L, 1L, 10.0),
			(11L, 1L, 11.0)
		)
		val _result = tensor.matricization(1, true)
		val result = _result.toCoordinateMatrix().entries.toDF()
		assert(_result.numCols() == 2)
		assert(_result.numRows() == 12)
		val resultSeq = result.collect().toSeq
		assert(result.distinct().collect().size == resultSeq.size)
		assert(resultSeq.size == dataCompareResult.size)
		for (row <- result.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
	}
	
	test("test transposed matricization mode 2") {
		val dataCompareResult = Seq[(Long, Long, Double)](
			(0L, 0L, 1.0),
			(1L, 0L, 2.0),
			(6L, 0L, 3.0),
			(7L, 0L, 4.0),
			(0L, 1L, 5.0),
			(1L, 1L, 6.0),
			(6L, 1L, 7.0),
			(7L, 1L, 8.0),
			(9L, 1L, 9.0),
			(10L, 1L, 10.0),
			(11L, 1L, 11.0)
		)
		val _result = tensor.matricization(2, true)
		val result = _result.toCoordinateMatrix().entries.toDF()
		assert(_result.numCols() == 2)
		assert(_result.numRows() == 12)
		val resultSeq = result.collect().toSeq
		assert(result.distinct().collect().size == resultSeq.size)
		assert(resultSeq.size == dataCompareResult.size)
		for (row <- result.distinct().collect().map(row => row.toSeq)) {
			assert(dataCompareResult.contains((row(0).asInstanceOf[Long], row(1).asInstanceOf[Long], row(2).asInstanceOf[Double])))
		}
	}
}
