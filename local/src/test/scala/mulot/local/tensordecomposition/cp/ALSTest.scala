package mulot.local.tensordecomposition.cp

import breeze.linalg.DenseMatrix
import mulot.local.Tensor
import org.scalatest.FunSuite

class ALSTest extends FunSuite {
	test("test CP ALS") {
		val file = getClass.getResource("/tensor_3_100_5clusters10.csv").getPath
		//val file = getClass.getResource("/tensor_3_100000_1.0E-9.csv").getPath
		val bufferedSource = io.Source.fromFile(file)
		val tensorData: Map[Array[_], Double] = (for (line <- bufferedSource.getLines.drop(1)) yield {
			val cols = line.split(",").map(_.trim)
			cols.dropRight(1).toArray[Object] -> cols(cols.size - 1).toDouble
		}).toMap
		bufferedSource.close
		val tensor = Tensor(tensorData, 3, Array("d0", "d1", "d2"))
		val rank = 1
		val begin = System.currentTimeMillis()
		val kruskal = ALS(tensor, rank).withComputeCorcondia(true).execute()
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		println(s"Corcondia = ${kruskal.corcondia}")
		for (matrix <- kruskal.A) {
			println(matrix)
			println()
		}
	}
	
	test("test CP from synthetic data") {
		val nbClusters = 3
		val clustersSize = 5
		var data = Map[Array[_], Double]()
		for (i <- 0 until nbClusters) {
			data ++= createCluster(i * clustersSize, clustersSize, 10.0)
		}
		
		val tensor = Tensor(data, 3, Array("d1", "d2", "d3"))
		val rank = nbClusters
		
		val begin = System.currentTimeMillis()
		//val kruskal = ALS(tensor, rank, ALS.NORM_L1, 25, computeCorcondia = true)
		val kruskal = ALS(tensor, rank).withComputeCorcondia(true).execute()
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		println(s"Corcondia = ${kruskal.corcondia}")
		var i: Int = 0
		for (matrix <- kruskal.A) {
			val matrix2 = DenseMatrix.zeros[Double](matrix.rows, matrix.cols)
			val indexes = tensor.inverseDimensionsIndex(i)
			matrix.foreachPair{case((a, b), v) => matrix2(indexes(a).toString.toInt, b) = v}
			println(matrix2)
			println()
			i += 1
		}
	}
	
	private def createCluster(beginIndex: Int, clusterSize: Int, value: Double = 1.0): Map[Array[Any], Double] = {
		(for (i <- beginIndex until (beginIndex + clusterSize);
			 j <- beginIndex until (beginIndex + clusterSize);
			 k <- beginIndex until (beginIndex + clusterSize)) yield {
			Array(i.toLong.asInstanceOf[Any], j.toLong, k.toLong) -> value
		}).toMap
	}
}
