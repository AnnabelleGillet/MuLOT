import breeze.linalg._

import mulot.local.Tensor
import mulot.local.tensordecomposition._
import mulot.local.tensordecomposition.cp.ALS
import mulot.local.tensordecomposition.cp.ALS._
import mulot.local.tensordecomposition.cp.CoupledALS
import mulot.local.tensordecomposition.cp.CoupledALS._
import mulot.core.tensordecomposition.CoupledDimension

import java.awt.Color
import collection.JavaConverters._

object ExecutionTime {
	def addData(nb: Int, dimension1: Int, dimension2: Int, dimension3: Int, value: Double = 10.0): Map[Array[Int], Double] = {
		val rand = new scala.util.Random
		(for (_ <- 0 until nb) yield {
			Array(rand.nextInt(dimension1), rand.nextInt(dimension2), rand.nextInt(dimension3)) -> (value + (rand.nextInt(6) - 3))
		}).toMap
	}

	// Execute with "scala -classpath lib/*:. ExecutionTime.scala"
	def main(args: Array[String]): Unit = {
		var result = List[String]()
		val nbTensors = Array(2, 3, 4)
		val dimensionsSize = Array(/*100,*/ 1000/*, 10000, 100000*/)
		val nbElements = Array(100, 1000, 10000, 100000, 1000000)
		for (t <- nbTensors) {
			for (size <- dimensionsSize) {
				for (nb <- nbElements) {
					// Experiment setup
					val data = addData(nb, size, size, size)
					val tensors = for (_ <- 0 until t) yield {
						Tensor.fromIndexedMap(data, 3, Array(size, size, size), Array("d1", "d2", "d3"))
					}
					
					val sharedDimensions = for (t1 <- 0 until t - 1) yield {
						CoupledDimension(tensors(t1), tensors(t1 + 1), Map[Int, Int](0 -> 0))
					}
					val coupledDecomposition = CoupledALS(tensors.toArray, 3, sharedDimensions.toArray).withMaxIterations(5)
					val times = (for (i <- 0 until 5) yield {
						val begin = System.currentTimeMillis()
						val resultCoupledDecomposition = coupledDecomposition.execute()
						val end = System.currentTimeMillis()
						end - begin
					}).toArray
					val time = times.sorted.tail.drop(1).reduce(_ + _).toDouble / 3
					result :+= s"$t tensors, $size dimensions' size, $nb elements: ${time}ms"
				}
			}
		}
		println(result.mkString("\n"))
	}
}
