import breeze.linalg._
import breeze.numerics.abs
import breeze.stats.distributions.Rand.FixedSeed.randBasis

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
	val APPROXIMATED_WEIGHTED_KENDALL_CORRELATION = CoupledCP.MergingScores.approximatedWeightedKendallCorrelation(0.5)
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

	// Execute with "scala -classpath lib/*:. ExecutionTime.scala"
	// Scala version used: 3.3.3
	def main(args: Array[String]): Unit = {
		val nbRepetitions = 5
		
		// Number of elements
		var nbElementsExecutionTimeWK = Array[Array[Double]]()
		var nbElementsExecutionTimeAWK = Array[Array[Double]]()
		for (nbElements <- List(10, 100, 1000, 10000, 100000/*, 1000000*/)) yield {
			val factorMatrices = (for (i <- 0 until 2) yield abs(DenseMatrix.rand(nbElements, 1, breeze.stats.distributions.Uniform(-24, -1)).mapValues(math.pow(10, _) * Math.random()))).toArray
			factorMatrices(0)(0 until 10, 0) := 1.0
			factorMatrices(1)(0 until 10, 0) := 1.0
			var time = (for (i <- 0 until nbRepetitions) yield {
				merging(factorMatrices, APPROXIMATED_WEIGHTED_KENDALL_CORRELATION)
			}).sum
			println(nbElements + ": " + (time.toDouble / nbRepetitions))
			nbElementsExecutionTimeAWK :+= Array(nbElements, time.toDouble / nbRepetitions)
			
			time = (for (i <- 0 until nbRepetitions) yield {
				merging(factorMatrices, WEIGHTED_KENDALL_CORRELATION)
			}).sum
			println(nbElements + ": " + (time.toDouble / nbRepetitions))
			nbElementsExecutionTimeWK :+= Array(nbElements, time.toDouble / nbRepetitions)
		}
		println("nb_elements = [" + nbElementsExecutionTimeAWK.map(_(0)).mkString(", ") + "]")
		println("nb_elements_wk = [" + nbElementsExecutionTimeWK.map(_(1)).mkString(", ") + "]")
		println("nb_elements_awk = [" + nbElementsExecutionTimeAWK.map(_(1)).mkString(", ") + "]")
		plotResult(nbElementsExecutionTimeWK, nbElementsExecutionTimeAWK, "", "Size of common dimension")
		
		// Number of merges
		var nbMergesExecutionTimeWK = Array[Array[Double]]()
		var nbMergesExecutionTimeAWK = Array[Array[Double]]()
		for (nbMerges <- 10 to 100 by 10) yield {
			val factorMatrices = (for (i <- 0 until 2) yield abs(DenseMatrix.rand(10000, if (i == 0) 1 else nbMerges, breeze.stats.distributions.Uniform(-24, -1)).mapValues(math.pow(10, _) * Math.random()))).toArray
			factorMatrices(0)(0 until 10, 0) := 1.0
			factorMatrices(1)(0 until 10, 0 until nbMerges) := 1.0
			var time = (for (i <- 0 until nbRepetitions) yield {
				merging(factorMatrices, APPROXIMATED_WEIGHTED_KENDALL_CORRELATION)
			}).sum
			println(nbMerges + ": " + (time.toDouble / nbRepetitions))
			nbMergesExecutionTimeAWK :+= Array(nbMerges, time.toDouble / nbRepetitions)
			
			time = (for (i <- 0 until nbRepetitions) yield {
				merging(factorMatrices, WEIGHTED_KENDALL_CORRELATION)
			}).sum
			println(nbMerges + ": " + (time.toDouble / nbRepetitions))
			nbMergesExecutionTimeWK :+= Array(nbMerges, time.toDouble / nbRepetitions)
		}
		println("nb_merges = [" + nbMergesExecutionTimeAWK.map(_(0)).mkString(", ") + "]")
		println("nb_merges_wk = [" + nbMergesExecutionTimeWK.map(_(1)).mkString(", ") + "]")
		println("nb_merges_awk = [" + nbMergesExecutionTimeAWK.map(_(1)).mkString(", ") + "]")
		plotResult(nbMergesExecutionTimeWK, nbMergesExecutionTimeAWK, "", "Number of merges")
		
		// Number of tensors
		var nbTensorsExecutionTimeWK = Array[Array[Double]]()
		var nbTensorsExecutionTimeAWK = Array[Array[Double]]()
		val nbTensorsExecutionTime = (for (nbTensors <- 5 to 50 by 5) yield {
			val factorMatrices = (for (i <- 0 until nbTensors) yield abs(DenseMatrix.rand(10000, 1, breeze.stats.distributions.Uniform(-24, -1)).mapValues(math.pow(10, _) * Math.random()))).toArray
			for (i <- 0 until nbTensors) factorMatrices(i)(0 until 10, 0) := 1.0
			var time = (for (i <- 0 until nbRepetitions) yield {
				merging(factorMatrices, APPROXIMATED_WEIGHTED_KENDALL_CORRELATION)
			}).sum
			println(nbTensors + ": " + (time.toDouble / nbRepetitions))
			nbTensorsExecutionTimeAWK :+= Array(nbTensors, time.toDouble / nbRepetitions)
			
			time = (for (i <- 0 until nbRepetitions) yield {
				merging(factorMatrices, WEIGHTED_KENDALL_CORRELATION)
			}).sum
			println(nbTensors + ": " + (time.toDouble / nbRepetitions))
			nbTensorsExecutionTimeWK :+= Array(nbTensors, time.toDouble / nbRepetitions)
		}).toArray
		println("nb_tensors = [" + nbTensorsExecutionTimeAWK.map(_(0)).mkString(", ") + "]")
		println("nb_tensors_wk = [" + nbTensorsExecutionTimeWK.map(_(1)).mkString(", ") + "]")
		println("nb_tensors_awk = [" + nbTensorsExecutionTimeAWK.map(_(1)).mkString(", ") + "]")
		plotResult(nbTensorsExecutionTimeWK, nbTensorsExecutionTimeAWK, "", "Number of tensors")
	}
	 
	/*
	 * Produce a visualisation for the metrics.
	 */
	def plotResult(dataWK: Array[Array[Double]], dataAWK: Array[Array[Double]], title: String, parameter: String): Unit = {
		val canvas = LinePlot.of(dataWK, Line.Style.SOLID, Color.BLUE, "Weighted Kendall correlation").canvas()
		canvas.add(LinePlot.of(dataAWK, Line.Style.SOLID, Color.RED, "Approximated weighted Kendall correlation"))
		canvas.setTitle(title)
		canvas.setAxisLabels(parameter, s"Time (ms)")
		canvas.window()
	}
}

