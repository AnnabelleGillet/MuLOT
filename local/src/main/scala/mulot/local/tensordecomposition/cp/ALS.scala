package mulot.local.tensordecomposition.cp

import breeze.linalg.svd.SVD
import breeze.linalg.{DenseMatrix, DenseVector, NotConvergedException, inv, max, min, pinv, sum, svd, svdr}
import breeze.numerics.abs
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import mulot.core.tensordecomposition.AbstractKruskal
import mulot.core.tensordecomposition.cp.Norms
import mulot.local.Tensor
import mulot.local.tensordecomposition.cp.ALS.Initializers
import scribe.Logging

object ALS extends Logging {
	def apply(tensor: Tensor, rank: Int): ALS = {
		new ALS(tensor, rank)
	}
	
	object Initializers {
		def gaussian(tensor: Tensor, rank: Int): Array[DenseMatrix[Double]] = {
			(for (i <- 0 until tensor.order) yield {
				val matrix = abs(DenseMatrix.rand(tensor.dimensionsSize(i), rank, breeze.stats.distributions.Gaussian(0.01, 1.0)))
				matrix /= max(matrix)
				matrix
			}).toArray
		}
		
		def hosvd(tensor: Tensor, rank: Int): Array[DenseMatrix[Double]] = {
			(for (i <- 0 until tensor.order) yield {
				logger.info(s"Compute HOSVD for dimension $i")
				val begin = System.currentTimeMillis()
				val result = svdr(tensor.matricization(i), rank).U
				logger.info(s"Dimension $i computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
				result
			}).toArray
		}
	}
	
	object ConvergenceMethods {
		/**
		 * The Factor Match Score (FMS) is used as convergence criteria to determine when to stop the iteration.
		 * It represents the similarity between the factor matrices of two iterations, with a value between 0 and 1 (at 0
		 * the matrices are completely different, and they are the same at 1). This function returns 1 minus the factor
		 * match score.
		 */
		def factorMatchScore(previousResult: AbstractKruskal[DenseMatrix[Double]], currentResult: AbstractKruskal[DenseMatrix[Double]]): Double = {
			val begin = System.currentTimeMillis()
			val fms = 1.0 - computeFactorMatchScore(currentResult.factorMatrices, currentResult.lambdas, previousResult.factorMatrices, previousResult.lambdas)
			logger.info(s"FMS = $fms, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			fms
		}
	}
	
	/**
	 * Compute the Factor Match Score for 2 sets of matrices.
	 *
	 * @param currentMatrices       the factor matrices of the current iteration
	 * @param currentLambdas        the lambdas of the current iteration
	 * @param lastIterationMatrices the factor matrices of the last iteration
	 * @param lastIterationLambdas  the lambdas of the last iteration
	 * @return the Factor Match Score, between 0 and 1
	 */
	private[mulot] def computeFactorMatchScore(currentMatrices: Array[DenseMatrix[Double]], currentLambdas: Array[Double],
								 lastIterationMatrices: Array[DenseMatrix[Double]], lastIterationLambdas: Array[Double]): Double = {
		val matricesMultiplied = for (i <- currentMatrices.indices) yield currentMatrices(i).t * lastIterationMatrices(i)
		val currentMatricesNorms = for (currentMatrix <- currentMatrices) yield {
			val norms = new Array[Double](currentLambdas.length)
			for (r1 <- norms.indices) {
				norms(r1) = 0.0
			}
			
			currentMatrix.foreachPair { case ((_, j), v) => norms(j) += v * v }
			for (r2 <- currentLambdas.indices) {
				norms(r2) = math.sqrt(norms(r2))
			}
			
			norms
		}
		val lastIterationMatricesNorms = for (lastIterationMatrix <- lastIterationMatrices) yield {
			val norms = new Array[Double](currentLambdas.length)
			for (r1 <- norms.indices) {
				norms(r1) = 0.0
			}
			
			lastIterationMatrix.foreachPair { case ((_, j), v) => norms(j) += v * v }
			for (r2 <- currentLambdas.indices) {
				norms(r2) = math.sqrt(norms(r2))
			}
			
			norms
		}
		
		var score = 0.0
		for (rank <- currentLambdas.indices) {
			val e1 = currentLambdas(rank) * currentMatricesNorms.aggregate(1.0)((v, l) => v * l(rank), (v1, v2) => v1 * v2)
			val e2 = lastIterationLambdas(rank) * lastIterationMatricesNorms.aggregate(1.0)((v, l) => v * l(rank), (v1, v2) => v1 * v2)
			val penalty = 1 - (math.abs(e1 - e2) / math.max(e1, e2))
			var tmpScore = 1.0
			for (i <- currentMatrices.indices) {
				tmpScore *= math.abs(matricesMultiplied(i)(rank, rank)) / (currentMatricesNorms(i)(rank) * lastIterationMatricesNorms(i)(rank))
			}
			score += penalty * tmpScore
		}
		score / currentLambdas.length
	}
	
	/**
	 * Compute the MTTKRP for the given tensor.
	 *
	 * @param tensorData           the [[Map]] containing the sparse data of the tensor
	 * @param currentDimension     the dimension concerning by the current iteration of the CP decomposition
	 * @param currentDimensionSize the size of the current dimension
	 * @param dimensionsSize       the size of all the dimensions of the tensor
	 * @param rank                 the rank of the CP decomposition
	 * @param factorMatrices       the current factor matrices of the CP decomposition
	 * @return the result of the MTTKRP as a [[DenseMatrix]]
	 */
	private[mulot] def computeMTTKRP(tensorData: Map[Array[Int], Double],
							  currentDimension: Int,
							  currentDimensionSize: Int,
							  dimensionsSize: Array[Int],
							  rank: Int,
							  factorMatrices: Array[DenseMatrix[Double]]): DenseMatrix[Double] = {
		val matrix = DenseMatrix.zeros[Double](currentDimensionSize, rank)
		val locks = Array.fill(currentDimensionSize) {
			new AnyRef
		}
		for ((keys, value) <- tensorData.par) {
			val currentDimensionIndex = keys(currentDimension)
			// Find the corresponding value in the corresponding block of the matrices of the Khatri Rao product
			val vectorValue = DenseVector.fill[Double](rank, value).t
			for (i <- dimensionsSize.indices if i != currentDimension) {
				val currentIndex = keys(i)
				vectorValue := vectorValue *:* factorMatrices(i)(currentIndex, ::)
			}
			
			locks(currentDimensionIndex).synchronized {
				matrix(currentDimensionIndex, ::) := matrix(currentDimensionIndex, ::) +:+ vectorValue
			}
		}
		matrix
	}
}

class ALS private(override var tensor: Tensor, val rank: Int)
	extends mulot.core.tensordecomposition.cp.ALS[Tensor, DenseMatrix[Double], Map[String, Array[Map[Any, Double]]]]
		with Logging {
	
	type Return = ALS
	
	private[mulot] var initializer: (Tensor, Int) => Array[DenseMatrix[Double]] = ALS.Initializers.gaussian
	override private[mulot] var convergenceMethod: (Kruskal, Kruskal) => Double = ALS.ConvergenceMethods.factorMatchScore
	
	override protected def internalCopy(): Return = {
		val newDecomposition = new ALS(tensor, rank)
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
	def withInitializer(initializer: (Tensor, Int) => Array[DenseMatrix[Double]]): Return = {
		val newObject = this.copy()
		newObject.initializer = initializer
		newObject
	}
	
	override protected def kruskalToExplicitValues(kruskal: Kruskal): Map[String, Array[Map[Any, Double]]] = {
		(for (i <- tensor.dimensionsName.indices) yield {
			val matrix = kruskal.A(i)
			val mapRanks = (for (r <- 0 until matrix.cols) yield {
				(for ((key, value) <- matrix(::, r).iterator) yield {
					tensor.inverseDimensionsIndex(i)(key) -> value
				}).toMap
			}).toArray
			tensor.dimensionsName(i) -> mapRanks
		}).toMap
	}
	
	override def execute(): Kruskal = {
		val tensorData = tensor.tensorIntegerData
		
		// Factor matrices initialization
		val factorMatrices = /*null +: */initializer(tensor, rank)
		var lastIterationFactorMatrices = new Array[DenseMatrix[Double]](tensor.order)

		// Lambda initialization
		val lambdas = new Array[Double](rank)
		var lastIterationLambdas = new Array[Double](rank)
		var lastIterationKruskal = Kruskal(factorMatrices, lambdas, None)
		// V is updated for each dimension rather than recalculated
		var v = (for (k <- 1 until factorMatrices.length) yield
			factorMatrices(k)).reduce((m1, m2) => (m1.t * m1) *:* (m2.t * m2))
		var convergence = false
		var nbIterations = 1
		while (!convergence) {
			val cpBegin = System.currentTimeMillis()
			logger.info(s"iteration $nbIterations")
			
			for (i <- 0 until tensor.order) {
				// Remove current dimension from V
				if (nbIterations > 1 || i > 0) {
					v = (v /:/ (factorMatrices(i).t * factorMatrices(i))).map(x => if (x.isNaN) 0.0 else x)
				}
				
				// MTTKRP
				val mttkrp = ALS.computeMTTKRP(tensorData, i, tensor.dimensionsSize(i), tensor.dimensionsSize, rank, factorMatrices)
				
				// pinverse
				factorMatrices(i) = mttkrp * pinv(v)
				if (nonNegativity) {
					factorMatrices(i) = factorMatrices(i).map(v => if (v < 0.0) 0.0 else v)
				}
				
				// Compute lambda
				for (j <- 0 until rank) {
					lambdas(j) = 0.0
				}
				if (norm == Norms.L2) {
					factorMatrices(i).foreachPair { case ((_, j), v) => lambdas(j) += v * v }
					for (j <- 0 until rank) {
						lambdas(j) = math.sqrt(lambdas(j))
					}
				} else {
					factorMatrices(i).foreachPair { case ((_, j), v) => lambdas(j) += math.abs(v) }
				}
				// Normalize factor matrices
				for (j <- 0 until rank) {
					factorMatrices(i)(::, j) := factorMatrices(i)(::, j) *:* (1 / lambdas(j))
				}
				
				// Update of V
				v = v *:* (factorMatrices(i).t * factorMatrices(i))
			}
			
			// Compute the Factor Match Score to see if the decomposition converges
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
			
			lastIterationFactorMatrices = for (m <- factorMatrices) yield m
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
		val corcondia = if (computeCorcondia) {
			val begin = System.currentTimeMillis()
			val corcondia = Some(computeCorcondiaScore(tensor, factorMatrices, lambdas))
			logger.info(s"CORCONDIA = ${corcondia.get}, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			corcondia
		} else None
		
		Kruskal(factorMatrices, lambdas, corcondia)
	}
	
	/**
	 * Compute the core consistency diagnostic (CORCONDIA), that gives a hint if the rank of the decomposition
	 * is adapted to the data. Will always yield 100 for a rank 1 decomposition.
	 *
	 * @param tensor the tensor on which the CP decomposition has been applied
	 * @param _matrices the factor matrices resulting of the application of the CP decomposition
	 * @param lambdas the lambdas resulting of the application of the CP decomposition
	 * @return the CORCONDIA score
	 */
	def computeCorcondiaScore(tensor: Tensor, _matrices: Array[DenseMatrix[Double]], lambdas: Array[Double]): Double = {
		val rank = lambdas.length
		
		val firstMatrix = _matrices(0).copy
		for (k <- 0 until rank) {
			firstMatrix(::, k) := firstMatrix(::, k) *:* lambdas(k)
		}
		val matrices = firstMatrix +: _matrices.tail
		
		val svds = for (matrix <- matrices) yield {
			try {
				val svdResult = svdr(matrix, rank)
				if (svdResult.∑.size < rank) {
					return Double.NaN
				}
				svdResult
			} catch {
				case _: NotConvergedException => {
					val SVD(u, s, vt) = svd(matrix)
					if (s.size < rank) {
						return Double.NaN
					}
					SVD(u(::, 0 until rank), s(0 until rank), vt(0 until rank, ::))
				}
			}
		}
		val uut = utt(tensor, (for (i <- svds.indices) yield svds(i).U.t).toArray, rank)
		
		def vtkronEuut(vector: DenseVector[Double], v: Array[DenseMatrix[Double]]): DenseVector[Double] = {
			val result = DenseVector.zeros[Double](vector.size)
			for (r <- 0 until vector.size) {
				val kronV = {
					var result = DenseVector.ones[Double](1)
					var index = r
					var matrixIndex = r % rank
					index -= matrixIndex
					index /= rank
					for (matrix <- v) {
						result = kronVectors(result, matrix(matrixIndex, ::).t)
						matrixIndex = index % rank
						index -= matrixIndex
						index /= rank
					}
					result
				}
				result(r) = sum(kronV *:* vector)
			}
			result
		}
		
		val kronE = {
			var result = DenseVector.ones[Double](1)
			for (svd <- svds) {
				result = kronVectors(result, svd.∑.map(x => math.pow(x, -1)))
			}
			result
		}
		
		val kronEuut = kronE *:* uut
		val gCore = vtkronEuut(kronEuut, (for (i <- svds.indices) yield svds(i).Vt.t).toArray)
		
		val currentIndexes = new Array[Int](matrices.length)
		for (i <- currentIndexes.indices) {
			currentIndexes(i) = 0
		}
		var sumCorcondia = 0.0
		var continue = true
		while (continue) {
			val diagonal = if (currentIndexes.forall(_ == currentIndexes.head)) 1.0 else 0.0
			
			var gIndex = 0
			var mul = 1
			for (i <- currentIndexes.indices.reverse) {
				gIndex += currentIndexes(i) * mul
				mul *= rank
			}
			
			sumCorcondia += math.pow(gCore(gIndex) - diagonal, 2)
			// Update indexes
			currentIndexes(matrices.length - 1) += 1
			var currentMatrix = matrices.length - 1
			while (currentMatrix >= 0 && currentIndexes(currentMatrix) >= rank - 1) {
				currentIndexes(currentMatrix) = 0
				currentMatrix -= 1
				if (currentMatrix < 0) {
					continue = false
				} else {
					currentIndexes(currentMatrix) += 1
				}
			}
		}
		100 * (1 - (sumCorcondia / rank))
	}
	
	private def utt(tensor: Tensor, u: Array[DenseMatrix[Double]], rank: Int): DenseVector[Double] = {
		val result: DenseVector[Double] = tensor.tensorIntegerData.par.aggregate(DenseVector.zeros[Double](math.pow(rank, u.length).toInt))((v, entry) => {
			// Find the corresponding index in each matrices of the khatri rao product
			val dimensionsIndex = for (i <- tensor.dimensionsSize.indices) yield {
				entry._1(i)
			}
			
			// Find the corresponding value in the corresponding block of the matrices of the khatri rao product
			var kronU = entry._2 * DenseVector.ones[Double](1)
			for (i <- dimensionsIndex.indices) {
				val currentIndex = dimensionsIndex(i)
				val currentMatrix = u(i)
				kronU = kronVectors(kronU, currentMatrix(::, currentIndex))
			}
			v := v +:+ kronU
			v
		}, _ +:+ _)
		result
	}
	
	private def kronVectors(a: DenseVector[Double], b: DenseVector[Double]): DenseVector[Double] = {
		val result: DenseVector[Double] = DenseVector.zeros[Double](a.size * b.size)
		for ((i, av) <- a.activeIterator) {
			result((i * b.size) until ((i + 1) * b.size)) := av * b
		}
		result
	}
}
