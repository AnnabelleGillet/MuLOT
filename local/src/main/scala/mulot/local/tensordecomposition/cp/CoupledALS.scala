package mulot.local.tensordecomposition.cp

import breeze.linalg.{DenseMatrix, inv, max, min}
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
		var referencingTensors = scala.collection.mutable.ArrayDeque.empty[Seq[(Tensor, Int)]]
		coupledDimensions.foreach(e => {
			assert(_tensors.contains(e.tensor1), s"Tensor ${e.tensor1} not in tensors.")
			assert(_tensors.contains(e.tensor2), s"Tensor ${e.tensor2} not in tensors.")
			for ((k, v) <- e.mapping) {
				var indexOfDimension = -1
				for (i <- 0 until nbDifferentCommonDimensions if indexOfDimension == -1) {
					if (referencingTensors(i).contains((e.tensor1, k)) ||
						referencingTensors(i).contains((e.tensor2, v))) {
						indexOfDimension = i
						if (!referencingTensors(i).contains((e.tensor1, k))) {
							referencingTensors(i) :+= (e.tensor1, k)
						}
						if (!referencingTensors(i).contains((e.tensor2, v))) {
							referencingTensors(i) :+= (e.tensor2, v)
						}
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
		def fixed(factorMatrices: Array[Array[DenseMatrix[Double]]])(tensors: Array[Tensor], rank: Int): Array[Array[DenseMatrix[Double]]] = {
			factorMatrices.clone()
		}
		
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
		def factorMatchScore(previousResult: AbstractKruskal[Array[DenseMatrix[Double]]], currentResult: AbstractKruskal[Array[DenseMatrix[Double]]], print: Boolean = true): Double = {
			val begin = System.currentTimeMillis()
			val fms = 1.0 - (for (i <- previousResult.factorMatrices.indices) yield {
				ALS.computeFactorMatchScore(currentResult.factorMatrices(i), currentResult.lambdas, previousResult.factorMatrices(i), previousResult.lambdas)
			}).sum / previousResult.factorMatrices.length
			if (print) {
				logger.info(s"FMS = $fms, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			}
			fms
		}
	}
}

/**
 * Implementation of the De Lathauwer algorithm, with an inner normalization step to take into consideration
 * tensors of different weight.
 *
 * @param tensors
 * @param rank
 * @param referencingTensors
 * @param commonDimensions
 */
class CoupledALS private(val tensors: Array[Tensor], override var rank: Int, val referencingTensors: Array[Seq[(Tensor, Int)]], val commonDimensions: Array[Map[Int, Int]]) extends mulot.core.tensordecomposition.cp.ALS[Tensor, Array[DenseMatrix[Double]], Array[Map[String, Array[Map[Any, Double]]]]]
		with Logging {
	type Return = CoupledALS
	
	override var tensor: Tensor = _
	private[mulot] var initializer: (Array[Tensor], Int) => Array[Array[DenseMatrix[Double]]] = CoupledALS.Initializers.gaussian
	override private[mulot] var convergenceMethod: (Kruskal, Kruskal, Boolean) => Double = CoupledALS.ConvergenceMethods.factorMatchScore
	
	override protected def internalCopy(): Return = {
		val newDecomposition = new CoupledALS(tensors, rank, referencingTensors, commonDimensions)
		newDecomposition
	}
	
	override private[mulot] def copy(): Return = {
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
						tensor.inverseDimensionsIndex(j)(key) -> value
					}).toMap
				}).toArray
				tensor.dimensionsName(j) -> mapRanks
			}).toMap
		}).toArray
	}
	
	override def execute(): Kruskal = {
		val tensorsData = for (t <- tensors) yield t.tensorIntegerData
		
		// Factor matrices initialization
		val factorMatrices = initializer(tensors, this.rank)
		/*val rank = if (factorMatrices(0).length != this.rank) {
			factorMatrices(0).length
		} else this.rank*/
		var lastIterationFactorMatrices = new Array[Array[DenseMatrix[Double]]](tensors.length)
		
		// Lambda initialization
		val lambdas = new Array[Double](rank)
		var lastIterationLambdas = new Array[Double](rank)
		var lastIterationKruskal = Kruskal(factorMatrices, lambdas, None)
		
		var convergence = false
		var nbIterations = 1
		val begin = System.currentTimeMillis()
		
		def internalIteration(tensorIndexes: Array[Int], dimensionIndexes: Array[Int]): Unit = {
			var factorMatrix = factorMatrices(tensorIndexes.head)(dimensionIndexes.head)
			val v = (for (k <- tensorIndexes.indices) yield {
				(for (l <- factorMatrices(k).indices if l != dimensionIndexes(k)) yield {
					factorMatrices(k)(l)
				}).reduce((m1, m2) => (m1.t * m1) *:* (m2.t * m2))
			}).reduce((m1, m2) => m1 +:+ m2)
			val vInv = inv(v)
			
			// MTTKRP
			for (j <- 0 until rank) {
				lambdas(j) = 0.0
			}
			factorMatrix = (for (i <- tensorIndexes.indices) yield {
				val matrix = ALS.computeMTTKRP(tensorsData(tensorIndexes(i)), dimensionIndexes(i), tensors(tensorIndexes(i)).dimensionsSize(dimensionIndexes(i)), tensors(tensorIndexes(i)).dimensionsSize, rank, factorMatrices(tensorIndexes(i)))
				matrix := (vInv * matrix.t).t
				val innerNormalization = new Array[Double](rank)
				if (norm == Norms.L2) {
					matrix.foreachPair { case ((_, j), v) => innerNormalization(j) += v * v }
				} else {
					matrix.foreachPair { case ((_, j), v) => innerNormalization(j) += abs(v) }
				}
				for (j <- 0 until rank) {
					innerNormalization(j) = math.sqrt(innerNormalization(j))
					lambdas(j) += innerNormalization(j) * (1 / tensorIndexes.length)
				}
				for (j <- 0 until rank) {
					matrix(::, j) := matrix(::, j) *:* (1 / innerNormalization(j))
				}
				matrix
			}).reduce(_ +:+ _)
			
			if (nonNegativity) {
				factorMatrix = factorMatrix.map(v => if (v < 0.0) 0.0 else v)
			}
			
			for (i <- tensorIndexes.indices) {
				factorMatrices(tensorIndexes(i))(dimensionIndexes(i)) = factorMatrix.copy
			}
		}
		
		val maxOrder = tensors.map(_.order).max
		while (!convergence) {
			val cpBegin = System.currentTimeMillis()
			if (nbIterations % printEvery == 0) {
				logger.info(s"iteration $nbIterations")
			}
			
			// Start with common dimensions
			for (i <- referencingTensors.indices) {
				internalIteration(referencingTensors(i).map(r => tensors.indexOf(r._1)).toArray, referencingTensors(i).map(_._2).toArray)
			}
			
			// Continue with all the other dimensions that are not shared
			for (o <- 0 until maxOrder; i <- tensors.indices; if o < tensors(i).order && !commonDimensions(i).contains(o)) {
				internalIteration(Array(i), Array(o))
			}
			
			// Fix signs
			for (r <- 0 until factorMatrices(0)(0).cols) {
				var toFlip = List[(Int, Int, Int)]()
				for (i <- factorMatrices.indices; j <- factorMatrices(i).indices) {
					val minValue = min(factorMatrices(i)(j)(::, r))
					val maxValue = max(factorMatrices(i)(j)(::, r))
					if (-minValue >= maxValue) {
						toFlip :+= (i, j, r)
					}
				}
				for (i <- toFlip.indices by 2) {
					if (i + 1 < toFlip.size) {
						val flip1 = toFlip(i)
						val flip2 = toFlip(i + 1)
						factorMatrices(flip1._1)(flip1._2)(::, flip1._3) := -factorMatrices(flip1._1)(flip1._2)(::, flip1._3)
						factorMatrices(flip2._1)(flip2._2)(::, flip2._3) := -factorMatrices(flip2._1)(flip2._2)(::, flip2._3)
					}
				}
			}
			
			// Compute the convergence score
			if (nbIterations > 1 && computeConvergence) {
				val currentKruskal = Kruskal(for (m <- factorMatrices) yield m, for (l <- lambdas) yield l, None)
				val convergenceScore = convergenceMethod(lastIterationKruskal, currentKruskal, nbIterations % printEvery == 0)
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
			
			if (nbIterations % printEvery == 0) {
				logger.info(s"iteration $nbIterations computed in ${(System.currentTimeMillis() - cpBegin).toDouble / 1000.0}s")
			}
			
			// Check if the iterations must stop
			if (nbIterations >= maxIterations) {
				convergence = true
			} else {
				nbIterations += 1
			}
		}
		
		logger.info(s"Coupled ALS computed in $nbIterations iterations (${(System.currentTimeMillis() - begin).toDouble / 1000.0}s)")
		
		// If required, compute CORCONDIA
		val corcondia = /*if (computeCorcondia) {
			begin = System.currentTimeMillis()
			val corcondia = Some(computeCorcondiaScore(tensor, factorMatrices, lambdas))
			logger.info(s"CORCONDIA = ${corcondia.get}, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			corcondia
		} else*/ None
		
		Kruskal(factorMatrices, lambdas, corcondia)
	}
}
