package mulot.local.tensordecomposition.cp

import breeze.linalg.{DenseMatrix, DenseVector}
import mulot.core.tensordecomposition.CoupledDimension
import mulot.local.Tensor
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers.{convertToAnyShouldWrapper, equal}

class CoupledCPTest extends AnyFunSuite {
	test("test Spearman's correlation perfect match") {
		val vectors = List(
			DenseVector(1.0, 2.0, 3.0, 4.0, 5.0),
			DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		)
		val mergedVector = DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		CoupledCP.MergingScores.spearmanCorrelation(mergedVector, vectors) should equal (1.0)
	}
	
	test("test Spearman's correlation inverse match") {
		val vectors = List(
			DenseVector(5.0, 4.0, 3.0, 2.0, 1.0),
			DenseVector(5.0, 4.0, 3.0, 2.0, 1.0)
		)
		val mergedVector = DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		CoupledCP.MergingScores.spearmanCorrelation(mergedVector, vectors) should equal (-1.0)
	}
	
	test("test Kendall's correlation perfect match") {
		val vectors = List(
			DenseVector(1.0, 2.0, 3.0, 4.0, 5.0),
			DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		)
		val mergedVector = DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		CoupledCP.MergingScores.kendallCorrelation(mergedVector, vectors) should equal(1.0)
	}
	
	test("test Kendall's correlation inverse match") {
		val vectors = List(
			DenseVector(5.0, 4.0, 3.0, 2.0, 1.0),
			DenseVector(5.0, 4.0, 3.0, 2.0, 1.0)
		)
		val mergedVector = DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		CoupledCP.MergingScores.kendallCorrelation(mergedVector, vectors) should equal(-1.0)
	}
	
	test("test weighted Kendall's correlation perfect match") {
		val vectors = List(
			DenseVector(1.0, 2.0, 3.0, 4.0, 5.0),
			DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		)
		val mergedVector = DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		CoupledCP.MergingScores.weightedKendallCorrelation(mergedVector, vectors) should equal(1.0)
	}
	
	test("test weighted Kendall's correlation inverse match") {
		val vectors = List(
			DenseVector(5.0, 4.0, 3.0, 2.0, 1.0),
			DenseVector(5.0, 4.0, 3.0, 2.0, 1.0)
		)
		val mergedVector = DenseVector(1.0, 2.0, 3.0, 4.0, 5.0)
		CoupledCP.MergingScores.weightedKendallCorrelation(mergedVector, vectors) should equal(-1.0)
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
				Array(i, j, k) -> (10.0)// + math.random() * 10)
			}).toMap
		}
		var data2 = Map[Array[Int], Double]()
		data2 = (for (i <- 0 until 15;
					  j <- 0 until 5;
					  k <- 0 until 5) yield {
			Array(i, j, k) -> (10.0)// + math.random() * 10)
		}).toMap
		data2 ++= (for (i <- 15 until 30;
						j <- 5 until 10;
						k <- 5 until 10) yield {
			Array(i, j, k) -> (10.0)// + math.random() * 10)
		}).toMap
		
		val tensor = Tensor.fromIndexedMap(data1, 3, Array(30, 30, 30), Array("d1", "d2", "d3"))
		val tensor2 = Tensor.fromIndexedMap(data2, 3, Array(30, 10, 10), Array("d0", "d1", "d2"))
		val cp1 = ALS(tensor, 3)
		val cp2 = ALS(tensor2, 2)
		val coupledALS = CoupledCP(Array(cp1, cp2), Array(0, 0))
		val begin = System.currentTimeMillis()
		val kruskal = coupledALS.execute()
		println(kruskal.A(0).mkString("\n\n"))
		println(kruskal.A(1).mkString("\n\n"))
		println(s"Computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
	}
}
