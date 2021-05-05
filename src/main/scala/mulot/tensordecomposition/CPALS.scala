package mulot.tensordecomposition

import mulot.Tensor
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix._
import org.apache.spark.sql.SparkSession

object CPALS {
	val NORM_L1 = "l1"
	val NORM_L2 = "l2"
	case class Kruskal(A: Array[ExtendedBlockMatrix], lambdas: Array[Double], corcondia: Option[Double])
	
	/**
	 * Computes the CAMDECOMP PARAFAC decomposition on the tensor.
	 */
	def compute(tensor: Tensor, rank: Int, norm: String = NORM_L1, maxIterations: Int = 25,
				minFms: Double = 0.99, highRank: Option[Boolean] = None, computeCorcondia: Boolean = false)
			   (implicit spark: SparkSession): Kruskal = {
		val nbColsPerBlock = 1024
		val useSparkPinv = if (highRank.isDefined) highRank.get else rank >= 100
		val tensorData = tensor.data.cache
		val factorMatrices = new Array[ExtendedBlockMatrix](tensor.order)
		var lambdas = new Array[Double](tensor.order)
		var lastIterationFactorMatrices = new Array[ExtendedBlockMatrix](tensor.order)
		var lastIterationLambdas = new Array[Double](tensor.order)
		var fms = 0.0
		
		// Randomized initialization
		for (i <- 1 until tensor.order) {
			factorMatrices(i) = ExtendedBlockMatrix.gaussian(tensor.dimensionsSize(i), rank)
		}
		// V is updated for each dimension rather than recalculated
		var v = (for (k <- 1 until factorMatrices.size) yield
			factorMatrices(k)).reduce((m1, m2) => (m1.transpose.multiply(m1)).hadamard(m2.transpose.multiply(m2)))
		var termination = false
		var nbIterations = 1
		while (!termination) {
			val cpBegin = System.currentTimeMillis()
			println("iteration " + nbIterations)
			for (i <- 0 until tensor.order) {
				// Remove current dimension from V
				if (factorMatrices(i) != null) {
					v = v.hadamard(factorMatrices(i).transpose.multiply(factorMatrices(i)), (m1, m2) => (m1 /:/ m2).toDenseMatrix.map(x => if (x.isNaN()) 0.0 else x))
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
				
				// Compute lambda
				if (norm == NORM_L2) {
					lambdas = factorMatrices(i).normL2()
				} else {
					lambdas = factorMatrices(i).normL1()
				}
				factorMatrices(i) = if (rank > nbColsPerBlock)
					factorMatrices(i).multiplyByArray(lambdas.map(l => 1 / l))
				else {
					factorMatrices(i).applyOperation(m => {
						for (k <- 0 until rank) {
							m(::, k) := m(::, k) *:* (1 / lambdas(k))
						}
						m
					})
				}
				
				// Update of V
				v = v.hadamard(factorMatrices(i).transpose.multiply(factorMatrices(i)))
			}
			
			// Compute the Factor Match Score to see if the decomposition converges
			if (nbIterations > 1) {
				val begin = System.currentTimeMillis()
				fms = ExtendedBlockMatrix.factorMatchScore(factorMatrices, lambdas, lastIterationFactorMatrices, lastIterationLambdas)
				println(s"FMS: $fms in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
			}
			
			lastIterationFactorMatrices = for (m <- factorMatrices) yield m
			lastIterationLambdas = for (l <- lambdas) yield l
			
			if (fms > minFms || nbIterations >= maxIterations) {
				termination = true
			} else {
				nbIterations += 1
			}
			println(s"CP in ${(System.currentTimeMillis() - cpBegin).toDouble / 1000.0}s")
		}
		
		var corcondia: Option[Double] = None
		if (computeCorcondia) {
			corcondia = Some(ExtendedBlockMatrix.corcondia(tensorData, tensor.dimensionsSize.toArray,
					factorMatrices(0).applyOperation(m => {
						for (k <- 0 until rank) {
							m(::, k) := m(::, k) *:* lambdas(k)
						}
						m
					}) +: factorMatrices.tail, rank, tensor.valueColumnName))
			println(s"CORCONDIA: $corcondia")
		}
		
		Kruskal(factorMatrices, lambdas, corcondia)
	}
}
