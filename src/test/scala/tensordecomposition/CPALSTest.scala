package tensordecomposition

import com.holdenkarau.spark.testing.SharedSparkContext
import org.apache.spark.sql.SparkSession
import org.scalatest.FunSuite

class CPALSTest extends FunSuite with SharedSparkContext {
	test("test CP ALS") {
		implicit val spark = SparkSession.builder().master("local[*]").getOrCreate()
		val file = getClass.getResource("/tensor_3_100_0.1.csv").getPath
		val tensor = Tensor.fromIndexedDataFrame(
			spark.read.option("header", true).csv(file),
			List(100L, 100L, 100L))
		
		val result = tensor.runCPALS(3, 2)
		for ((k, v) <- result) {
			//println(k)
			//v.show(300)
			assert(v.select("rank").distinct.count() == 3)
			assert(v.select(k).distinct.count() == 100)
		}
	}
}
