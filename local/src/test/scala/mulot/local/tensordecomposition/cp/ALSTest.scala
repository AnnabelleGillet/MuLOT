package mulot.local.tensordecomposition.cp

import breeze.linalg.DenseMatrix
import mulot.core.tensordecomposition.CoupledDimension
import mulot.local.Tensor
import org.scalatest.funsuite.AnyFunSuite

class ALSTest extends AnyFunSuite {
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
		val rank = 3
		val begin = System.currentTimeMillis()
		val kruskal = ALS(tensor, rank).withComputeCorcondia(true).execute()
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		println(s"Corcondia = ${kruskal.corcondia}")
		for (matrix <- kruskal.A) {
			println(matrix)
			println()
		}
	}
	
	test("test coupled CP ALS") {
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
		val coupledALS = CoupledALS(Array(tensor, tensor), rank, Array(CoupledDimension(tensor, tensor, Map(0 -> 0))))
		val begin = System.currentTimeMillis()
		val kruskal = coupledALS.execute()
		println(kruskal.A(0).mkString("\n"))
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
	}
	
	test("test coupled CP ALS with synthetic clusters") {
		var data1 = Map[Array[Int], Double]()
		for (c <- 0 until 3) {
			data1 ++= (for (i <- c * 10 until (c * 10 + 10);
							j <- c * 10 until (c * 10 + 10);
							k <- c * 10 until (c * 10 + 10)) yield {
				Array(i, j, k) -> 10.0
			}).toMap
		}
		var data2 = Map[Array[Int], Double]()
		data2 = (for (i <- 0 until 15;
						  j <- 0 until 5;
						  k <- 0 until 5) yield {
			Array(i, j, k) -> 10.0
		}).toMap
		data2 ++= (for (i <- 15 until 30;
						  j <- 5 until 10;
						  k <- 5 until 10) yield {
			Array(i, j, k) -> 10.0
		}).toMap
		
		val tensor = Tensor.fromIndexedMap(data1, 3, Array(30, 30, 30), Array("d1", "d2", "d3"))
		val tensor2 = Tensor.fromIndexedMap(data2, 3, Array(30, 10, 10), Array("d0", "d1", "d2"))
		val rank = 3
		val coupledALS = CoupledALS(Array(tensor, tensor2), rank, Array(CoupledDimension(tensor, tensor2, Map(0 -> 0))))
		val begin = System.currentTimeMillis()
		val kruskal = coupledALS.execute()
		println(kruskal.A(0).mkString("\n\n"))
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
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
