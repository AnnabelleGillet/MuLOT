package mulot.distributed.tensordecomposition.cp

import mulot.core.tensordecomposition.CoupledDimension
import mulot.distributed.Tensor
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.funsuite.AnyFunSuite

class ALSTest extends AnyFunSuite {
	test("test CP ALS") {
		
		implicit val spark = SparkSession.builder().master("local[*]").getOrCreate()
		spark.sparkContext.setLogLevel("WARN")
		
		val size = 100L
		val file = getClass.getResource("/tensor_3_100_0.1.csv").getPath
		val tensor = Tensor.fromIndexedDataFrame(
			spark.read.option("header", true).csv(file).dropDuplicates("d0", "d1", "d2"),
			Array(size, size, size))
		val rank = 3
		val begin = System.currentTimeMillis()
		val cp = ALS(tensor, rank).withComputeCorcondia(true).withMaxIterations(5).withConvergenceThreshold(0.5)
		
		val result = cp.execute().toExplicitValues()
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		
		for ((k, v) <- result) {
			v.show()
			assert(v.select("rank").distinct.count() == rank)
			println(v.select(k).distinct().count())
			assert(v.select(k).distinct.count() == size)
		}
	}
	
	test("test coupled CP ALS") {
		implicit val spark = SparkSession.builder().master("local[*]").getOrCreate()
		spark.sparkContext.setLogLevel("WARN")
		
		val size = 100L
		val file = getClass.getResource("/tensor_3_100_0.001_5clusters10.csv").getPath
		val tensor = Tensor.fromIndexedDataFrame(
			spark.read.option("header", true).csv(file).dropDuplicates("d0", "d1", "d2"),
			Array(size, size, size))
		val rank = 3
		val begin = System.currentTimeMillis()
		
		val coupledALS = CoupledALS(Array(tensor, tensor), rank, Array(CoupledDimension(tensor, tensor, Map(0 -> 0)))).withMaxIterations(2)
		
		val result = coupledALS.execute()
		for ((k, v) <- result.toExplicitValues()(0)) {
			//v.show(300)
			assert(v.select("rank").distinct.count() == rank)
			println(v.select(k).distinct().count())
			assert(v.select(k).distinct.count() == size)
		}
		
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
	}
}
