package mulot.local.tensordecomposition.tucker

import breeze.linalg.{CSCMatrix, DenseMatrix, DenseVector, NotConvergedException, inv, max, min, pinv, sum, svd, svdr}
import breeze.numerics.abs
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import mulot.core.tensordecomposition.{AbstractHOOIResult, tucker}
import mulot.local.Tensor
import scribe.Logging
import scala.collection.parallel.CollectionConverters._
object HOOI extends Logging {
	def apply(tensor: Tensor, ranks: Array[Int]): HOOI = {
		new HOOI(tensor, ranks)
	}
	
	object Initializers {
		def gaussian(tensor: Tensor, ranks: Array[Int]): Array[DenseMatrix[Double]] = {
			(for (i <- 0 until tensor.order) yield {
				val matrix = abs(DenseMatrix.rand(tensor.dimensionsSize(i), ranks(i), breeze.stats.distributions.Gaussian(0.01, 1.0)))
				matrix /= max(matrix)
				matrix
			}).toArray
		}
		
		def hosvd(tensor: Tensor, ranks: Array[Int]): Array[DenseMatrix[Double]] = {
			(for (i <- 0 until tensor.order) yield {
				logger.info(s"Compute HOSVD for dimension $i")
				val begin = System.currentTimeMillis()
				val result = svdr(tensor.matricization(i), ranks(i)).U
				logger.info(s"Dimension $i computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
				result
			}).toArray
		}
	}
	
	object ConvergenceMethods {
		/**
		 * The Frobenius norm is used as convergence criteria to determine when to stop the iteration.
		 * It represents the similarity between the core tensors of two iterations, with a value between 0 and 1 (at 0
		 * the core tensors are completely different, and they are the same at 1).
		 */
		def frobeniusNormOnCoreTensor(originalTensor: Tensor): (AbstractHOOIResult[DenseMatrix[Double], Tensor], AbstractHOOIResult[DenseMatrix[Double], Tensor], Boolean) => Double = {
			val originalFrobenius = originalTensor.frobeniusNorm()
			
			def compute(currentResult: AbstractHOOIResult[DenseMatrix[Double], Tensor], previousResult: AbstractHOOIResult[DenseMatrix[Double], Tensor], print: Boolean = true): Double = {
				val begin = System.currentTimeMillis()
				val frobenius = currentResult.coreTensor.frobeniusNorm()
				val residualNorm = math.sqrt(originalFrobenius * originalFrobenius - frobenius * frobenius)
				val frobeniusDifference = 1 - ({
					if (residualNorm.isNaN) 0.0 else residualNorm
				} / originalFrobenius)
				
				val previousFrobenius = previousResult.coreTensor.frobeniusNorm()
				val previousResidualNorm = math.sqrt(originalFrobenius * originalFrobenius - previousFrobenius * previousFrobenius)
				val previousFrobeniusDifference = 1 - ({
					if (previousResidualNorm.isNaN) 0.0 else previousResidualNorm
				} / originalFrobenius)
				
				val score = math.abs(previousFrobeniusDifference - frobeniusDifference)
				if (print) {
					logger.info(s"Frobenius on core tensor = $score, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
				}
				score
			}
			
			compute
		}
	}
}

class HOOI private[tucker](override var tensor: Tensor, val ranks: Array[Int])
	extends tucker.HOOI[Tensor, DenseMatrix[Double], Map[String, Array[Map[Any, Double]]]]
		with Logging {
	
	type Return = HOOI
	
	override private[mulot] var initializer: (Tensor, Array[Int]) => Array[DenseMatrix[Double]] = HOOI.Initializers.hosvd
	override private[mulot] var convergenceMethod: (HOOIResult, HOOIResult, Boolean) => Double = HOOI.ConvergenceMethods.frobeniusNormOnCoreTensor(tensor)
	
	override protected def internalCopy(): Return = {
		val newDecomposition = new HOOI(tensor, ranks)
		newDecomposition
	}
	
	override private[mulot] def copy(): Return = {
		val newDecomposition = super.copy()
		newDecomposition
	}
	
	override protected def resultToExplicitValues(result: HOOIResult): Map[String, Array[Map[Any, Double]]] = {
		(for (i <- tensor.dimensionsName.indices) yield {
			val matrix = result.U(i)
			val mapRanks = (for (r <- 0 until matrix.cols) yield {
				(for ((key, value) <- matrix(::, r).iterator) yield {
					tensor.inverseDimensionsIndex(i)(key) -> value
				}).toMap
			}).toArray
			tensor.dimensionsName(i) -> mapRanks
		}).toMap
	}
	
	override def execute(): HOOIResult = {
		// Initialisation
		val begin = System.currentTimeMillis()
		val factorMatrices = initializer(tensor, ranks)
		
		// Order the dimensions of the tensor to start from the biggest one,
		// so we can reduce quickly the total size of the tensor
		val dimensionsOrder = tensor.dimensionsSize.zipWithIndex.sortWith((v1, v2) => v1._1 >= v2._1).map(v => v._2)
		var convergence = false
		var finalCoreTensor: Tensor = null
		//var lastIterationHOOIResult: HOOIResult = null
		var lastIterationCoreTensor: Tensor = null
		var iteration = 1
		
		// Iterate while the convergence criteria is not met
		while (!convergence && iteration <= maxIterations) {
			if (iteration % printEvery == 0) {
				logger.info(s"Iteration $iteration")
			}
			val tuckerBegin = System.currentTimeMillis()
			var previousCoreTensor = tensor.tensorIntegerData.toList
			// Compute the new factor matrices
			for (dimensionIndice <- dimensionsOrder.indices) {
				val dimension = dimensionsOrder(dimensionIndice)
				
				// Prepare the core tensor for the iteration
				var coreTensor = previousCoreTensor
				// Compute the core tensor with mode-n product except with the factor matrix of the current dimension
				for (i <- (dimensionIndice + 1) until tensor.order) {
					val currentDimension = dimensionsOrder(i)
					coreTensor = modeNProduct(coreTensor, ranks(currentDimension), currentDimension, factorMatrices(currentDimension), iteration % printEvery == 0)
				}
				
				// Compute the new factor matrix for the current dimension
				//factorMatrices(dimension) = ExtendedIndexedRowMatrix.fromIndexedRowMatrix(coreTensor.matricization(dimension, true)).VofSVD(ranks(dimension))
				//factorMatrices(dimension) = svdr(coreTensor.matricization(dimension), ranks(dimension)).U
				factorMatrices(dimension) = svdr(matricization(coreTensor, dimension, tensor, ranks), ranks(dimension)).U
				
				// Update the global core tensor
				previousCoreTensor = modeNProduct(previousCoreTensor, ranks(dimension), dimension, factorMatrices(dimension), iteration % printEvery == 0)
			}
			
			// Compute the Frobenius norm of the difference of the current core tensor and of the core tensor
			// of the previous iteration
			val _coreTensor = Tensor.fromIndexedMap(
				previousCoreTensor.toMap,
				tensor.order,
				ranks,
				tensor.dimensionsName
			)
			if (computeConvergence && iteration > 1) {
				val currentResult = HOOIResult(factorMatrices, _coreTensor)
				//val score = convergenceMethod(currentResult, lastIterationHOOIResult, iteration % printEvery == 0)
				val score = convergenceMethod(currentResult, HOOIResult(factorMatrices, lastIterationCoreTensor), iteration % printEvery == 0)
				if (score <= convergenceThreshold) {
					convergence = true
				}
			}
			
			//lastIterationHOOIResult = HOOIResult(factorMatrices, _coreTensor)
			lastIterationCoreTensor = _coreTensor
			
			// Keep the final core tensor if the convergence criteria is met
			if (convergence || iteration >= maxIterations) {
				finalCoreTensor = Tensor.fromIndexedMap(
					previousCoreTensor.toMap,
					tensor.order,
					ranks,
					tensor.dimensionsName
				)
			}
			iteration += 1
			
			if (iteration % printEvery == 0) {
				logger.info(s"Iteration $iteration computed in ${(System.currentTimeMillis() - tuckerBegin).toDouble / 1000.0}s")
			}
		}
		if (finalCoreTensor == null) {
			finalCoreTensor = tensor
			for (dimensionIndice <- dimensionsOrder.indices) {
				val dimension = dimensionsOrder(dimensionIndice)
				val newFinalCoreTensorData = modeNProduct(finalCoreTensor.tensorIntegerData.toList, tensor.dimensionsSize(dimension), dimension, factorMatrices(dimension), iteration % printEvery == 0)
				finalCoreTensor = Tensor.fromIndexedMap(
					newFinalCoreTensorData.toMap,
					tensor.order,
					ranks,
					tensor.dimensionsName
				)
			}
		}
		
		logger.info(s"HOOI computed in $iteration iterations (${(System.currentTimeMillis() - begin).toDouble / 1000.0}s)")
		
		HOOIResult(factorMatrices, finalCoreTensor)
	}
	
	private def matricization(data: List[(Array[Int], Double)], n: Int, tensor: Tensor, ranks: Array[Int]): DenseMatrix[Double] = {
		val newDimensionSize = (for (r <- ranks.indices if r != n) yield ranks(r)).product
		val matrix = DenseMatrix.zeros[Double](tensor.dimensionsSize(n), newDimensionSize)
		
		for ((k, v) <- data.par) {
			var j = 0
			var coef = 1
			for (i <- 0 until tensor.order if i != n) {
				j += k(i) * coef
				coef *= ranks(i)
			}
			matrix(k(n), j) = v
		}
		matrix
	}
	
	private def modeNProduct(tensor: List[(Array[Int], Double)], dimensionSize: Int, dimension: Int, factorMatrix: DenseMatrix[Double], print: Boolean): List[(Array[Int], Double)] = {
		val begin = System.currentTimeMillis()
		val newTensorData = tensor.par.groupBy(e => (for (i <- e._1.indices) yield if (i != dimension) e._1(i) else 0).toList).flatMap(e => {
			for (i <- 0 until dimensionSize) yield {
				val newKey = for (j <- e._1.indices) yield {
					if (j == dimension) {
						i
					} else {
						e._1(j)
					}
				}
				val newValue = for (k <- e._2.seq.indices) yield e._2(k)._2 * factorMatrix(e._2(k)._1(dimension), i)
				newKey.toArray -> newValue.sum
			}
		}).toList
		if (print) {
			logger.info(s"Mode-n product computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		}
		newTensorData
	}
}