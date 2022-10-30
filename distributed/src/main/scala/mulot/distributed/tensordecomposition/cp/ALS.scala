package mulot.distributed.tensordecomposition.cp

import mulot.core.tensordecomposition.cp
import mulot.core.tensordecomposition.cp.Norms
import mulot.distributed.Tensor
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix.fromBlockMatrix
import org.apache.spark.sql.{DataFrame, SparkSession}
import scribe.Logging

object ALS {
	def apply(tensor: Tensor, rank: Int)(implicit spark: SparkSession): ALS =  {
		new ALS(tensor, rank)(spark)
	}
}

class ALS private(val tensor: Tensor, val rank: Int)(implicit spark: SparkSession)
	extends cp.ALS[Tensor, ExtendedBlockMatrix, Map[String, DataFrame]]
		with Logging {
	
	object Initializers {
		def gaussian(tensor: Tensor, rank: Int): Array[ExtendedBlockMatrix] = {
			(for (i <- 1 until tensor.order) yield {
				ExtendedBlockMatrix.gaussian(tensor.dimensionsSize(i), rank)
			}).toArray
		}
		
		def hosvd(tensor: Tensor, rank: Int): Array[ExtendedBlockMatrix] = {
			(for (i <- 1 until tensor.order) yield {
				ExtendedBlockMatrix.hosvd(tensor, i, rank)
			}).toArray
		}
	}
	
	protected var highRank: Option[Boolean] = None
	override var initializer: (Tensor, Int) => Array[ExtendedBlockMatrix] = Initializers.gaussian
	
	override protected def copy(): ALS = {
		val newObject = new ALS(tensor, rank)
		newObject.highRank = this.highRank
		newObject
	}
	
	override protected def kruskalToExplicitValues(kruskal: Kruskal): Map[String, DataFrame] = {
		(for (i <- tensor.dimensionsName.indices) yield {
			var df = spark.createDataFrame(kruskal.A(i).toCoordinateMatrixWithZeros().entries).toDF("dimIndex", "rank", "val")
			if (tensor.dimensionsIndex.isDefined) {
				df = df.join(tensor.dimensionsIndex.get(i), "dimIndex").select("dimValue", "rank", "val")
				df = df.withColumnRenamed("dimValue", tensor.dimensionsName(i))
			} else {
				df = df.withColumnRenamed("dimIndex", tensor.dimensionsName(i))
			}
			tensor.dimensionsName(i) -> df
		}).toMap
	}
	
	def withHighRank(highRank: Boolean): this.type = {
		this.highRank = Some(highRank)
		this
	}
	
	override def execute(): Kruskal = {
		val nbColsPerBlock = 1024
		val useSparkPinv = if (highRank.isDefined) highRank.get else rank >= 100
		val tensorData = tensor.data.cache
		val factorMatrices = null +: initializer(tensor, rank)
		var lambdas = new Array[Double](tensor.order)
		var lastIterationFactorMatrices = new Array[ExtendedBlockMatrix](tensor.order)
		var lastIterationLambdas = new Array[Double](tensor.order)
		var fms = 0.0
		
		for (i <- 1 until tensor.order) {
			factorMatrices(i) = ExtendedBlockMatrix.gaussian(tensor.dimensionsSize(i), rank)
		}
		
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
			
			// Compute the Factor Match Score to see if the decomposition converges
			if (nbIterations > 1) {
				val begin = System.currentTimeMillis()
				fms = ExtendedBlockMatrix.factorMatchScore(factorMatrices, lambdas, lastIterationFactorMatrices, lastIterationLambdas)
				logger.info(s"FMS = $fms, computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			}
			
			lastIterationFactorMatrices = for (m <- factorMatrices) yield m
			lastIterationLambdas = for (l <- lambdas) yield l
			
			if (fms > minFms || nbIterations >= maxIterations) {
				convergence = true
			} else {
				nbIterations += 1
			}
			logger.info(s"iteration $nbIterations computed in ${(System.currentTimeMillis() - cpBegin).toDouble / 1000.0}s")
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
