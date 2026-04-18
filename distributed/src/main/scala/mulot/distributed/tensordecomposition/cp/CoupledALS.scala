package mulot.distributed.tensordecomposition.cp

import mulot.core.tensordecomposition.cp.Norms
import mulot.core.tensordecomposition.{AbstractKruskal, CoupledDimension}
import mulot.distributed.Tensor
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix.fromBlockMatrix
import org.apache.spark.sql.{DataFrame, SparkSession}
import scribe.Logging

object CoupledALS extends Logging {
	def apply(_tensors: Array[Tensor], rank: Int, coupledDimensions: Array[CoupledDimension[Tensor]])(implicit spark: SparkSession): CoupledALS = {
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
		val tensors: Array[Tensor] = (for (i <- _tensors.indices) yield {
			var newTensor = _tensors(i)
			for (commonDimension <- commonDimensions(i)) {
				newTensor = newTensor.reindex(commonDimension._1, newIndexes(commonDimension._2))
			}
			newTensor
		}).toArray
		
		new CoupledALS(tensors, rank, referencingTensors.map(_.map(e => (tensors(_tensors.indexOf(e._1)), e._2))).toArray, commonDimensions)
	}
	
	object Initializers {
		def gaussian(tensors: Array[Tensor], rank: Int)(implicit spark: SparkSession): Array[Array[ExtendedBlockMatrix]] = {
			for (tensor <- tensors) yield {
				ALS.Initializers.gaussian(tensor, rank)
			}
		}
		
		def hosvd(tensors: Array[Tensor], rank: Int)(implicit spark: SparkSession): Array[Array[ExtendedBlockMatrix]] = {
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
		def factorMatchScore(previousResult: AbstractKruskal[Array[ExtendedBlockMatrix]], currentResult: AbstractKruskal[Array[ExtendedBlockMatrix]], print: Boolean = true): Double = {
			val begin = System.currentTimeMillis()
			val fms = 1.0 - (for (i <- previousResult.factorMatrices.indices) yield {
				ExtendedBlockMatrix.factorMatchScore(currentResult.factorMatrices(i), currentResult.lambdas, previousResult.factorMatrices(i), previousResult.lambdas)
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
 * @param spark
 */
class CoupledALS private(val tensors: Array[Tensor], override var rank: Int, val referencingTensors: Array[Seq[(Tensor, Int)]], val commonDimensions: Array[Map[Int, Int]])
						(implicit spark: SparkSession)
	extends mulot.core.tensordecomposition.cp.ALS[Tensor, Array[ExtendedBlockMatrix], Array[Map[String, DataFrame]]]
	with Logging {
	type Return = CoupledALS
	
	override var tensor: Tensor = _
	private[mulot] var highRank: Option[Boolean] = None
	private[mulot] var initializer: (Array[Tensor], Int) => Array[Array[ExtendedBlockMatrix]] = CoupledALS.Initializers.gaussian
	override private[mulot] var convergenceMethod: (Kruskal, Kruskal, Boolean) => Double = CoupledALS.ConvergenceMethods.factorMatchScore
	
	override protected def internalCopy(): Return = {
		val newDecomposition = new CoupledALS(tensors, rank, referencingTensors, commonDimensions)
		newDecomposition
	}
	
	override private[mulot] def copy(): Return = {
		val newDecomposition = super.copy()
		newDecomposition.initializer = this.initializer
		newDecomposition.highRank = this.highRank
		newDecomposition
	}
	
	/**
	 * Choose which method to use to initialize the factor matrices.
	 *
	 * @param initializer the method to use
	 */
	def withInitializer(initializer: (Array[Tensor], Int) => Array[Array[ExtendedBlockMatrix]]): Return = {
		val newObject = this.copy()
		newObject.initializer = initializer
		newObject
	}
	
	override protected def kruskalToExplicitValues(kruskal: Kruskal): Array[Map[String, DataFrame]] = {
		(for (i <- tensors.indices) yield {
			val tensor = tensors(i)
			(for (j <- tensor.dimensionsName.indices) yield {
				var df = spark.createDataFrame(kruskal.A(i)(j).toCoordinateMatrixWithZeros().entries).toDF("dimIndex", "rank", "val")
				df = df.join(tensor.dimensionsIndex(i), "dimIndex").select("dimValue", "rank", "val")
				df = df.withColumnRenamed("dimValue", tensor.dimensionsName(i))
				tensor.dimensionsName(i) -> df
			}).toMap
		}).toArray
	}
	
	def withHighRank(highRank: Boolean): Return = {
		val newDecomposition = this.copy()
		newDecomposition.highRank = Some(highRank)
		newDecomposition
	}
	
	override def execute(): Kruskal = {
		val nbColsPerBlock = 1024
		
		val tensorsData = for (t <- tensors) yield t.data.cache()
		
		// Factor matrices initialization
		val factorMatrices = initializer(tensors, rank)
		var lastIterationFactorMatrices = new Array[Array[ExtendedBlockMatrix]](tensors.length)
		
		// Lambda initialization
		var lambdas = new Array[Double](rank)
		var lastIterationLambdas = new Array[Double](rank)
		var lastIterationKruskal = Kruskal(factorMatrices, lambdas, None)
		
		var convergence = false
		var nbIterations = 1
		var begin = System.currentTimeMillis()
		
		def internalIteration(tensorIndexes: Array[Int], dimensionIndexes: Array[Int]): Unit = {
			var factorMatrix = factorMatrices(tensorIndexes.head)(dimensionIndexes.head)
			val v = (for (k <- tensorIndexes.indices) yield {
				(for (l <- factorMatrices(k).indices if l != dimensionIndexes(k)) yield {
					factorMatrices(k)(l)
				}).reduce((m1, m2) => (m1.transpose.multiply(m1)).hadamard(m2.transpose.multiply(m2)))
			}).reduce((m1, m2) => m1.add(m2))
			val vInv = v.inverse()
			
			// MTTKRP
			for (j <- 0 until rank) {
				lambdas(j) = 0.0
			}
			factorMatrix = (for (i <- tensorIndexes.indices) yield {
				var matrix = if (rank > nbColsPerBlock) {
					ExtendedBlockMatrix.mttkrpHighRankDataFrame(tensorsData(i),
						(for (k <- factorMatrices(tensorIndexes(i)).indices if dimensionIndexes(i) != k) yield factorMatrices(tensorIndexes(i))(k)).toArray,
						(for (k <- tensors(tensorIndexes(i)).dimensionsSize.indices if dimensionIndexes(i) != k) yield tensors(tensorIndexes(i)).dimensionsSize(k)).toArray,
						dimensionIndexes(i),
						tensors(tensorIndexes(i)).dimensionsSize(dimensionIndexes(i)),
						rank,
						tensors(tensorIndexes(i)).valueColumnName
					)
				} else {
					ExtendedBlockMatrix.mttkrpDataFrame(tensorsData(i),
						(for (k <- factorMatrices(tensorIndexes(i)).indices if dimensionIndexes(i) != k) yield factorMatrices(tensorIndexes(i))(k)).toArray,
						(for (k <- tensors(tensorIndexes(i)).dimensionsSize.indices if dimensionIndexes(i) != k) yield tensors(tensorIndexes(i)).dimensionsSize(k)).toArray,
						dimensionIndexes(i),
						tensors(tensorIndexes(i)).dimensionsSize(dimensionIndexes(i)),
						rank,
						tensors(tensorIndexes(i)).valueColumnName
					)
				}
				matrix = (vInv.multiply(matrix.transpose)).transpose
				val innerNormalization = if (norm == Norms.L2) {
					factorMatrix.normL2()
				} else {
					factorMatrix.normL1()
				}
				matrix = matrix.multiplyByArray(innerNormalization.map(1.0 / _))
				for (j <- 0 until rank) {
					lambdas(j) += innerNormalization(j) * (1 / tensorIndexes.length)
				}
				matrix
			}).reduce(_.add(_))
			
			if (nonNegativity) {
				factorMatrix = ExtendedBlockMatrix.fromBreeze(factorMatrix.toSparseBreeze().map(v => if (v < 0.0) 0.0 else v))
			}
			
			for (i <- tensorIndexes.indices) {
				factorMatrices(tensorIndexes(i))(dimensionIndexes(i)) = factorMatrix
			}
		}
		
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
			for (i <- tensors.indices; j <- 0 until tensors(i).order if !commonDimensions(i).contains(j)) {
				internalIteration(Array(i), Array(j))
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
			
			lastIterationFactorMatrices = for (m <- factorMatrices) yield {for (n <- m) yield n}
			lastIterationLambdas = for (l <- lambdas) yield l
			
			if (nbIterations % printEvery == 0) {
				logger.info(s"iteration $nbIterations computed in ${(System.currentTimeMillis() - cpBegin).toDouble / 1000.0}s")
			}
			
			if (nbIterations >= maxIterations) {
				convergence = true
			} else {
				nbIterations += 1
			}
		}
		logger.info(s"Coupled ALS computed in $nbIterations iterations (${(System.currentTimeMillis() - begin).toDouble / 1000.0}s)")
		var corcondia: Option[Double] = None
		Kruskal(factorMatrices, lambdas, corcondia)
	}
}
