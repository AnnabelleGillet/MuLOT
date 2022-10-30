package mulot.distributed.tensordecomposition.cp

import mulot.distributed.Tensor
import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite

class ALSTest extends FunSuite {
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
		val cp = ALS(tensor, rank).withComputeCorcondia(true).withMaxIterations(10)
		
		val result = cp.execute().toExplicitValues()
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		for ((k, v) <- result) {
			assert(v.select("rank").distinct.count() == rank)
			println(v.select(k).distinct().count())
			assert(v.select(k).distinct.count() == size)
		}
	}
}
