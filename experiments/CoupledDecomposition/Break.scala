import breeze.linalg._
import breeze.numerics.abs
import breeze.stats.distributions.Rand.FixedSeed.randBasis

import java.io.ByteArrayOutputStream
import java.io.PrintStream

import mulot.local.Tensor
import mulot.local.tensordecomposition._
import mulot.local.tensordecomposition.cp.ALS
import mulot.local.tensordecomposition.cp.ALS._
import mulot.local.tensordecomposition.cp.CoupledALS
import mulot.local.tensordecomposition.cp.CoupledALS._
import mulot.local.tensordecomposition.cp.CoupledCP
import mulot.core.tensordecomposition.CoupledDimension
import mulot.core.tensordecomposition.cp._

import java.awt.Color
import collection.JavaConverters._

import smile.data._
import smile.data.`type`._
import smile.plot.swing._

object MergingScore {
	val WEIGHTED_KENDALL_CORRELATION = CoupledCP.MergingScores.weightedKendallCorrelation
	val APPROXIMATED_WEIGHTED_KENDALL_CORRELATION = CoupledCP.MergingScores.approximatedWeightedKendallCorrelation(0.5, true)
	/**
	 * Merging function.
	 */
	def merging(factorMatrices: Array[DenseMatrix[Double]], mergingScore: (DenseVector[Double], List[DenseVector[Double]]) => Double): Long = {
		val begin = System.currentTimeMillis()
		val ranksAdvance = Array.fill[Int](factorMatrices.length){0}
		var vectorsList = List[(Array[Int], DenseVector[Double])]()
		var over = false
		while (!over) {
			// Get factors of individual decompositions
			val factorsToMerge = (for (i <- ranksAdvance.indices) yield {
				var vector = factorMatrices(i)(::, ranksAdvance(i))
				// Normalize vector
				vector / max(vector)
			}).toList
			
			// Merge vector
			var mergedVector = factorsToMerge.reduce(min(_, _)).mapValues(x => if (x.isNaN) 0.0 else x)
			
			// Compute score
			val score = mergingScore(mergedVector, factorsToMerge)
			
			if (score > 0.15) {
				vectorsList :+= (ranksAdvance.clone(), mergedVector)
			}
			
			// Update ranks advance
			var i = 0
			var ok = false
			while (!ok && i < factorMatrices.length) {
				ranksAdvance(i) += 1
				if (ranksAdvance(i) >= factorMatrices(i).cols) {
					ranksAdvance(i) = 0
					i += 1
				} else {
					ok = true
				}
			}
			if (!ok) {
				over = true
			}
		}
		System.currentTimeMillis() - begin
	}

	// Execute with "scala -classpath lib/*:. Break.scala"
	// Scala version used: 3.3.3
	def main(args: Array[String]): Unit = {
		val originalOut = System.out
		val outputBuffer = new ByteArrayOutputStream()
		System.setOut(new PrintStream(outputBuffer))
		
		// Number of elements
		var executionTimeAWK = Array[Array[Double]]()
		val maxElements = 1000
		val step = 100
		for (nbElements <- step to maxElements by step) yield {
			for (nbElementsInCommon <- math.max(0, nbElements - (maxElements - nbElements)) to nbElements by step) yield {
				val factorMatrices = (for (i <- 0 until 2) yield abs(DenseMatrix.rand(maxElements, 1, breeze.stats.distributions.Uniform(-24, -2)).mapValues(math.pow(10, _) * Math.random()))).toArray
				factorMatrices(0)(0 until nbElements, 0) := Math.random()
				factorMatrices(1)((nbElements - nbElementsInCommon) until ((nbElements - nbElementsInCommon) + nbElements), 0) := Math.random()
				System.out.println(s"Nb: $nbElements $nbElementsInCommon")
				var time = merging(factorMatrices, APPROXIMATED_WEIGHTED_KENDALL_CORRELATION)
				executionTimeAWK :+= Array(nbElements, nbElementsInCommon, time.toDouble)
			}
		}
		
		System.setOut(originalOut)

		val interceptedLogs: String = outputBuffer.toString("UTF-8")
		val cleanedLogs = interceptedLogs.split("\n").filter(e => e.trim.startsWith("Break") || e.trim.startsWith("Nb"))

		println(s"Result:\n${cleanedLogs.mkString("\n")}")
		
		println("nb_elements = [" + executionTimeAWK.map(_(0)).mkString(", ") + "]")
		println("nb_elements_in_common = [" + executionTimeAWK.map(_(1)).mkString(", ") + "]")
		println("execution_time = [" + executionTimeAWK.map(_(2)).mkString(", ") + "]")
	}
}

