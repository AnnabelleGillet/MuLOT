package mulot.distributed.tensordecomposition.tucker

import mulot.distributed.Tensor
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers.{convertNumericToPlusOrMinusWrapper, convertToAnyShouldWrapper}
class HOOITest extends AnyFunSuite {
	implicit val spark: SparkSession = SparkSession.builder().master("local[*]").getOrCreate()
	spark.sparkContext.setLogLevel("WARN")
	
	test("test Tucker from synthetic data") {
		val nbClusters = 3
		val clustersSize = 5
		val size = nbClusters * clustersSize
		val sizes = Array[Long](size, size + clustersSize, size + clustersSize)
		var data = Seq[(Long, Long, Long, Double)]()
		for (i <- 0 until nbClusters) {
			data ++= createCluster(i * clustersSize, clustersSize, 10.0)
		}
		println("Data created")
		import spark.implicits._
		val ranks = Array(3, 3, 3)
		val valueColumnName = "val"
		val tensor = Tensor.fromIndexedDataFrame(
			data.toDF("d0", "d1", "d2", valueColumnName).select(valueColumnName, "d0", "d1", "d2"),
			sizes,
			valueColumnName = valueColumnName)
		println("Tensor created")
		val begin = System.currentTimeMillis()
		val result = HOOI(tensor, ranks).withMaxIterations(5).execute()
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		//result.coreTensor.data.orderBy(desc(valueColumnName)).show(50)
		// Check if the size of the core tensor is correct
		result.coreTensor.data.count() shouldBe ranks.product
		
		for (dimension <- 0 until 3) {
			// Check if all the dimensions of the factor matrix are correct
			result.U(dimension).numCols() shouldBe ranks(dimension)
			result.U(dimension).numRows() shouldBe sizes(dimension)
			val matrix = result.U(dimension).toBlockMatrix().toLocalMatrix()
			for (cluster <- 0 until nbClusters) {
				val firstVector = cluster * clustersSize
				for (vectorIndex <- 1 until clustersSize) {
					var norm = 0.0
					for (i <- 0 until matrix.numCols) {
						norm += math.pow(matrix(firstVector, i) + matrix(firstVector + vectorIndex, i), 2)
					}
					norm = math.sqrt(norm)
					norm shouldBe 0.0 +- 2.0
				}
			}
		}
	}
	
	private def createCluster(beginIndex: Int, clusterSize: Int, value: Double = 1.0): Seq[(Long, Long, Long, Double)] = {
		for (i <- beginIndex until (beginIndex + clusterSize);
			 j <- beginIndex until (beginIndex + clusterSize);
			 k <- beginIndex until (beginIndex + clusterSize)) yield {
			(i.toLong, j.toLong, k.toLong, value)
		}
	}
}
