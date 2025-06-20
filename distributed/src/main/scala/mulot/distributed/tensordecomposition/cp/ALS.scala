package mulot.distributed.tensordecomposition.cp

import mulot.core.tensordecomposition.{AbstractKruskal, DecompositionResult, cp}
import mulot.core.tensordecomposition.cp.Norms
import mulot.distributed.Tensor
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix.fromBlockMatrix
import org.apache.spark.sql.{DataFrame, SparkSession}
import scribe.Logging

object ALS extends Logging {
	def apply(tensor: Tensor, rank: Int)(implicit spark: SparkSession): ALS =  {
		new ALS(tensor, rank)(spark)
	}
	
	object Initializers {
		def gaussian(tensor: Tensor, rank: Int)(implicit spark: SparkSession): Array[ExtendedBlockMatrix] = {
			(for (i <- 0 until tensor.order) yield {
				ExtendedBlockMatrix.gaussian(tensor.dimensionsSize(i), rank)
			}).toArray
		}
		
		def hosvd(tensor: Tensor, rank: Int)(implicit spark: SparkSession): Array[ExtendedBlockMatrix] = {
			(for (i <- 0 until tensor.order) yield {
				ExtendedBlockMatrix.hosvd(tensor, i, rank)
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
		def factorMatchScore(previousResult: AbstractKruskal[ExtendedBlockMatrix], currentResult: AbstractKruskal[ExtendedBlockMatrix]): Double = {
			val begin = System.currentTimeMillis()
			val fms = 1.0 - ExtendedBlockMatrix.factorMatchScore(currentResult.factorMatrices, currentResult.lambdas, previousResult.factorMatrices, previousResult.lambdas)
			logger.info(s"FMS = $fms, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			fms
		}
	}
}

class ALS private(override var tensor: Tensor, val rank: Int)(implicit spark: SparkSession)
	extends cp.ALS[Tensor, ExtendedBlockMatrix, Map[String, DataFrame]]
		with Logging {
	
	override type Return = ALS
	
	private[mulot] var highRank: Option[Boolean] = None
	private[mulot] var initializer: (Tensor, Int) => Array[ExtendedBlockMatrix] = ALS.Initializers.gaussian
	
	override private[mulot] var convergenceMethod: (Kruskal, Kruskal) => Double = ALS.ConvergenceMethods.factorMatchScore
	
	override protected def internalCopy(): ALS = {
		val newDecomposition = new ALS(tensor, rank)
		newDecomposition
	}
	override protected def copy(): Return = {
		val newDecomposition = super.copy()
		newDecomposition.highRank = this.highRank
		newDecomposition.initializer = this.initializer
		newDecomposition
	}
	
	/**
	 * Choose which method to use to initialize the factor matrices.
	 *
	 * @param initializer the method to use
	 */
	def withInitializer(initializer: (Tensor, Int) => Array[ExtendedBlockMatrix]): Return = {
		val newObject = this.copy()
		newObject.initializer = initializer
		newObject
	}
	
	override protected def kruskalToExplicitValues(kruskal: Kruskal): Map[String, DataFrame] = {
		(for (i <- tensor.dimensionsName.indices) yield {
			var df = spark.createDataFrame(kruskal.A(i).toCoordinateMatrixWithZeros().entries).toDF("dimIndex", "rank", "val")
			df = df.join(tensor.dimensionsIndex(i), "dimIndex").select("dimValue", "rank", "val")
			df = df.withColumnRenamed("dimValue", tensor.dimensionsName(i))
			tensor.dimensionsName(i) -> df
		}).toMap
	}
	
	def withHighRank(highRank: Boolean): Return = {
		val newDecomposition = this.copy()
		newDecomposition.highRank = Some(highRank)
		newDecomposition
	}
	
	override def execute(): Kruskal = {
		val nbColsPerBlock = 1024
		val useSparkPinv = if (highRank.isDefined) highRank.get else rank >= 100
		val tensorData = tensor.data.cache
		val factorMatrices = /*null +:*/ initializer(tensor, rank)
		var lambdas = new Array[Double](tensor.order)
		var lastIterationFactorMatrices = new Array[ExtendedBlockMatrix](tensor.order)
		var lastIterationLambdas = new Array[Double](tensor.order)
		var lastIterationKruskal = Kruskal(lastIterationFactorMatrices, lastIterationLambdas, None)
		
		// V is updated for each dimension rather than recalculated
		var v = (for (k <- 1 until factorMatrices.length) yield
			factorMatrices(k)).reduce((m1, m2) => (m1.transpose.multiply(m1)).hadamard(m2.transpose.multiply(m2)))
		var convergence = false
		var nbIterations = 1
		while (!convergence) {
			val cpBegin = System.currentTimeMillis()
			logger.info(s"iteration $nbIterations")
			for (i <- 0 until tensor.order) {
				// Remove current dimension from V
				if (factorMatrices(i) != null) {
					v = v.hadamard(factorMatrices(i).transpose.multiply(factorMatrices(i)), (m1, m2) => (m1 /:/ m2).toDenseMatrix.map(x => if (x.isNaN) 0.0 else x))
				}
				// Compute MTTKRP
				val mttkrp = if (rank > nbColsPerBlock) {
					ExtendedBlockMatrix.mttkrpHighRankDataFrame(tensorData,
						(for (k <- factorMatrices.indices if i != k) yield factorMatrices(k)).toArray,
						(for (k <- tensor.dimensionsSize.indices if i != k) yield tensor.dimensionsSize(k)).toArray,
						i,
						tensor.dimensionsSize(i),
						rank,
						tensor.valueColumnName
					)
				} else {
					ExtendedBlockMatrix.mttkrpDataFrame(tensorData,
						(for (k <- factorMatrices.indices if i != k) yield factorMatrices(k)).toArray,
						(for (k <- tensor.dimensionsSize.indices if i != k) yield tensor.dimensionsSize(k)).toArray,
						i,
						tensor.dimensionsSize(i),
						rank,
						tensor.valueColumnName
					)
				}
				val pinv = if (useSparkPinv) v.sparkPinverse() else v.pinverse()
				factorMatrices(i) = mttkrp.multiply(pinv)
				if (nonNegativity) {
					var newFactorMatrix = factorMatrices(i).toSparseBreeze()
					newFactorMatrix = newFactorMatrix.map(v => if (v < 0.0) 0.0 else v)
					factorMatrices(i) = ExtendedBlockMatrix.fromBreeze(newFactorMatrix)
				}
				
				// Compute lambdas
				if (norm == Norms.L2) {
					lambdas = factorMatrices(i).normL2()
				} else {
					lambdas = factorMatrices(i).normL1()
				}
				factorMatrices(i) = if (rank > nbColsPerBlock)
										factorMatrices(i).multiplyByArray(lambdas.map(l => 1 / l))
									else {
										factorMatrices(i).divideByLambdas(lambdas)
									}
				
				// Update of V
				v = v.hadamard(factorMatrices(i).transpose.multiply(factorMatrices(i)))
			}
			
			// Compute the convergence score
			if (nbIterations > 1 && computeConvergence) {
				val currentKruskal = Kruskal(for (m <- factorMatrices) yield m, for (l <- lambdas) yield l, None)
				val convergenceScore = convergenceMethod(lastIterationKruskal, currentKruskal)
				if (convergenceScore <= convergenceThreshold) {
					convergence = true
				}
			}
			
			lastIterationFactorMatrices = for (m <- factorMatrices) yield m
			lastIterationLambdas = for (l <- lambdas) yield l
			lastIterationKruskal = Kruskal(for (m <- factorMatrices) yield m, for (l <- lambdas) yield l, None)
			
			logger.info(s"iteration $nbIterations computed in ${(System.currentTimeMillis() - cpBegin).toDouble / 1000.0}s")
			
			if (nbIterations >= maxIterations) {
				convergence = true
			} else {
				nbIterations += 1
			}
		}
		
		var corcondia: Option[Double] = None
		if (computeCorcondia) {
			val begin = System.currentTimeMillis()
			corcondia = Some(ExtendedBlockMatrix.corcondia(tensorData, tensor.dimensionsSize,
				factorMatrices(0).divideByLambdas(lambdas) +: factorMatrices.tail, rank, tensor.valueColumnName))
			logger.info(s"CORCONDIA = ${corcondia.get}, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		}
		
		Kruskal(factorMatrices, lambdas, corcondia)
	}
}
