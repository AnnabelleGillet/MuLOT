package mulot.local.tensordecomposition.cp

import breeze.linalg.{DenseMatrix, inv}
import breeze.numerics.abs
import mulot.core.tensordecomposition.{AbstractKruskal, CoupledDimension}
import mulot.core.tensordecomposition.cp.Norms
import mulot.local.Tensor
import scribe.Logging

object CoupledALS extends Logging {
	def apply(_tensors: Array[Tensor], rank: Int, coupledDimensions: Array[CoupledDimension[Tensor]]): CoupledALS = {
		val commonDimensions = new Array[Map[Int, Int]](_tensors.length)
		for (i <- commonDimensions.indices) {
			commonDimensions(i) = Map.empty[Int, Int]
		}
		
		var nbDifferentCommonDimensions = 0
		var referencingTensors = scala.collection.mutable.MutableList.empty[Seq[(Tensor, Int)]]
		coupledDimensions.foreach(e => {
			assert(_tensors.contains(e.tensor1), s"Tensor ${e.tensor1} not in tensors.")
			assert(_tensors.contains(e.tensor2), s"Tensor ${e.tensor2} not in tensors.")
			for ((k, v) <- e.mapping) {
				var indexOfDimension = -1
				for (i <- 0 until nbDifferentCommonDimensions if indexOfDimension == -1) {
					if (referencingTensors(i).contains((e.tensor1, k)) ||
						referencingTensors(i).contains((e.tensor2, v))) {
						indexOfDimension = i
						referencingTensors(i) :+= (e.tensor1, k)
						referencingTensors(i) :+= (e.tensor2, v)
					}
				}
				if (indexOfDimension == -1) {
					indexOfDimension = nbDifferentCommonDimensions
					referencingTensors :+= Seq((e.tensor1, k), (e.tensor2, v))
					nbDifferentCommonDimensions += 1
				}
				commonDimensions(_tensors.indexOf(e.tensor1)) += k -> indexOfDimension
				commonDimensions(_tensors.indexOf(e.tensor2)) += v -> indexOfDimension
			}
		})
		
		val newIndexes = for (rt <- referencingTensors) yield Tensor.reindexDimension(rt.toArray)
		val newInverseIndexes = for (newIndex <- newIndexes) yield newIndex.map(e => e._2 -> e._1)
		val tensors: Array[Tensor] = (for (i <- _tensors.indices) yield {
			var newTensor = _tensors(i)
			for (commonDimension <- commonDimensions(i)) {
				newTensor = newTensor.reindex(commonDimension._1, newIndexes(commonDimension._2), newInverseIndexes(commonDimension._2))
			}
			newTensor
		}).toArray
		
		new CoupledALS(tensors, rank, referencingTensors.map(_.map(e => (tensors(_tensors.indexOf(e._1)), e._2))).toArray, commonDimensions)
	}
	
	object Initializers {
		def gaussian(tensors: Array[Tensor], rank: Int): Array[Array[DenseMatrix[Double]]] = {
			for (tensor <- tensors) yield {
				ALS.Initializers.gaussian(tensor, rank)
			}
		}
		
		def hosvd(tensors: Array[Tensor], rank: Int): Array[Array[DenseMatrix[Double]]] = {
			for (tensor <- tensors) yield {
				ALS.Initializers.hosvd(tensor, rank)
			}
		}
	}
	
	object ConvergenceMethods {
		/**
		 * The Factor Match Score (FMS) is used as convergence criteria to determine when to stop the iteration.
		 * It represents the similarity between the factor matrices of two iterations, with a value between 0 and 1 (at 0
		 * the matrices are completely different, and they are the same at 1). This function returns 1 minus the factor
		 * match score.
		 */
		def factorMatchScore(previousResult: AbstractKruskal[Array[DenseMatrix[Double]]], currentResult: AbstractKruskal[Array[DenseMatrix[Double]]]): Double = {
			val begin = System.currentTimeMillis()
			val fms = 1.0 - (for (i <- previousResult.factorMatrices.indices) yield {
				ALS.computeFactorMatchScore(currentResult.factorMatrices(i), currentResult.lambdas, previousResult.factorMatrices(i), previousResult.lambdas)
			}).sum / previousResult.factorMatrices.length
			logger.info(s"FMS = $fms, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			fms
		}
	}
}

class CoupledALS private(val tensors: Array[Tensor], override val rank: Int, val referencingTensors: Array[Seq[(Tensor, Int)]], val commonDimensions: Array[Map[Int, Int]]) extends mulot.core.tensordecomposition.cp.ALS[Tensor, Array[DenseMatrix[Double]], Array[Map[String, Array[Map[Any, Double]]]]]
		with Logging {
	type Return = CoupledALS
	
	override var tensor: Tensor = _
	private[mulot] var initializer: (Array[Tensor], Int) => Array[Array[DenseMatrix[Double]]] = CoupledALS.Initializers.gaussian
	override private[mulot] var convergenceMethod: (Kruskal, Kruskal) => Double = CoupledALS.ConvergenceMethods.factorMatchScore
	
	override protected def internalCopy(): Return = {
		val newDecomposition = new CoupledALS(tensors, rank, referencingTensors, commonDimensions)
		newDecomposition
	}
	
	override protected def copy(): Return = {
		val newDecomposition = super.copy()
		newDecomposition.initializer = this.initializer
		newDecomposition
	}
	
	/**
	 * Choose which method to use to initialize the factor matrices.
	 *
	 * @param initializer the method to use
	 */
	def withInitializer(initializer: (Array[Tensor], Int) => Array[Array[DenseMatrix[Double]]]): Return = {
		val newObject = this.copy()
		newObject.initializer = initializer
		newObject
	}
	
	override protected def kruskalToExplicitValues(kruskal: Kruskal): Array[Map[String, Array[Map[Any, Double]]]] = {
		(for (i <- tensors.indices) yield {
			val tensor = tensors(i)
			(for (j <- tensor.dimensionsName.indices) yield {
				val matrix = kruskal.A(i)(j)
				val mapRanks = (for (r <- 0 until matrix.cols) yield {
					(for ((key, value) <- matrix(::, r).iterator) yield {
						tensor.inverseDimensionsIndex(i)(key) -> value
					}).toMap
				}).toArray
				tensor.dimensionsName(i) -> mapRanks
			}).toMap
		}).toArray
	}
	
	override def execute(): Kruskal = {
		val tensorsData = for (t <- tensors) yield t.tensorIntegerData
		
		// Factor matrices initialization
		val factorMatrices = initializer(tensors, rank)
		var lastIterationFactorMatrices = new Array[Array[DenseMatrix[Double]]](tensors.length)
		
		// Lambda initialization
		val lambdas = new Array[Double](rank)
		var lastIterationLambdas = new Array[Double](rank)
		var lastIterationKruskal = Kruskal(factorMatrices, lambdas, None)
		
		var convergence = false
		var nbIterations = 1
		
		def internalIteration(tensorIndexes: Array[Int], dimensionIndexes: Array[Int]): Unit = {
			var factorMatrix = factorMatrices(tensorIndexes.head)(dimensionIndexes.head)
			val v = (for (k <- tensorIndexes.indices) yield {
				(for (l <- factorMatrices(k).indices if l != dimensionIndexes(k)) yield {
					factorMatrices(k)(l)
				}).reduce((m1, m2) => (m1.t * m1) *:* (m2.t * m2))
			}).reduce((m1, m2) => m1 +:+ m2)
			
			// MTTKRP
			var mttkrp = ALS.computeMTTKRP(tensorsData(tensorIndexes.head), dimensionIndexes.head, tensors(tensorIndexes.head).dimensionsSize(dimensionIndexes.head), tensors(tensorIndexes.head).dimensionsSize, rank, factorMatrices(tensorIndexes.head))
			for (i <- tensorIndexes.indices.tail) {
				mttkrp = mttkrp +:+ ALS.computeMTTKRP(tensorsData(tensorIndexes(i)), dimensionIndexes(i), tensors(tensorIndexes(i)).dimensionsSize(dimensionIndexes(i)), tensors(tensorIndexes(i)).dimensionsSize, rank, factorMatrices(tensorIndexes(i)))
			}
			
			factorMatrix = (inv(v) * mttkrp.t).t
			if (nonNegativity) {
				factorMatrix = factorMatrix.map(v => if (v < 0.0) 0.0 else v)
			}
			
			// Compute lambda
			for (j <- 0 until rank) {
				lambdas(j) = 0.0
			}
			if (norm == Norms.L2) {
				factorMatrix.foreachPair { case ((_, j), v) => lambdas(j) += v * v }
				for (j <- 0 until rank) {
					lambdas(j) = math.sqrt(lambdas(j))
				}
			} else {
				factorMatrix.foreachPair { case ((_, j), v) => lambdas(j) += math.abs(v) }
			}
			// Normalize factor matrices
			for (j <- 0 until rank) {
				factorMatrix(::, j) := factorMatrix(::, j) *:* (1 / lambdas(j))
			}
			
			for (i <- tensorIndexes.indices) {
				factorMatrices(tensorIndexes(i))(dimensionIndexes(i)) = factorMatrix
			}
		}
		
		while (!convergence) {
			val cpBegin = System.currentTimeMillis()
			logger.info(s"iteration $nbIterations")
			
			// Start with common dimensions
			for (i <- referencingTensors.indices) {
				internalIteration(referencingTensors(i).map(r => tensors.indexOf(r._1)).toArray, referencingTensors(i).map(_._2).toArray)
			}
			
			// Continue with all the other dimensions that are not shared
			for (i <- tensors.indices; j <- 0 until tensors(i).order if !commonDimensions(i).contains(j)) {
				internalIteration(Array(i), Array(j))
			}
			
			// Compute the convergence score
			if (nbIterations > 1 && computeConvergence) {
				val currentKruskal = Kruskal(for (m <- factorMatrices) yield m, for (l <- lambdas) yield l, None)
				val convergenceScore = convergenceMethod(lastIterationKruskal, currentKruskal)
				if (convergenceScore <= convergenceThreshold) {
					convergence = true
				}
				lastIterationKruskal = currentKruskal
			} else {
				lastIterationKruskal = Kruskal(for (m <- factorMatrices) yield m, for (l <- lambdas) yield l, None)
			}
			
			lastIterationFactorMatrices = for (m <- factorMatrices) yield {
				for (n <- m) yield n
			}
			lastIterationLambdas = for (l <- lambdas) yield l
			
			logger.info(s"iteration $nbIterations computed in ${(System.currentTimeMillis() - cpBegin).toDouble / 1000.0}s")
			
			// Check if the iterations must stop
			if (nbIterations >= maxIterations) {
				convergence = true
			} else {
				nbIterations += 1
			}
		}
		
		// If required, compute CORCONDIA
		val corcondia = /*if (computeCorcondia) {
			val begin = System.currentTimeMillis()
			val corcondia = Some(computeCorcondiaScore(tensor, factorMatrices, lambdas))
			logger.info(s"CORCONDIA = ${corcondia.get}, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			corcondia
		} else*/ None
		
		Kruskal(factorMatrices, lambdas, corcondia)
	}
}
