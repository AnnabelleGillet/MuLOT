package mulot.distributed

import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite

class TensorTest extends FunSuite {
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
