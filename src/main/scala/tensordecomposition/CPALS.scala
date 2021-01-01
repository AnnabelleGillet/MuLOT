package tensordecomposition

import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix
import org.apache.spark.mllib.linalg.distributed.ExtendedBlockMatrix._
import org.apache.spark.sql.SparkSession

object CPALS {
	case class Kruskal(A: Array[ExtendedBlockMatrix], lambdas: Array[Double])
	
	/**
	 * Computes the CAMDECOMP PARAFAC decomposition on the tensor.
	 */
	def computeSparkCPALS(tensor: Tensor, rank: Int, maxIterations: Int = 25, checkpoint: Boolean = false)
						 (implicit spark: SparkSession): Kruskal = {
		val tensorMatricized = tensor.matricization()
		if (checkpoint) {
			tensorMatricized.foreach(m => {
				val mc = m.cache()
				mc.checkpoint()
				mc
			})
		}
		val result = new Array[ExtendedBlockMatrix](tensor.order)
		var lambda = new Array[Double](tensor.order)
		// Randomized initialization
		for (i <- 1 until tensor.order) {
			result(i) = ExtendedBlockMatrix.gaussian(tensor.dimensionsSize(i), rank)
		}
		// V is updated for each dimension rather than recalculated
		var v = (for (k <- 1 until result.size) yield
			result(k)).reduce((m1, m2) => (m1.transpose.multiply(m1)).hadamard(m2.transpose.multiply(m2)))
		var termination = false
		var nbIterations = 0
		while (!termination) {
			println("iteration " + nbIterations)
			for (i <- 0 until tensor.order) {
				// Remove current dimension from V
				if (result(i) != null) {
					v = v.hadamard(result(i).transpose.multiply(result(i)), (m1, m2) => m1 /:/ m2)
				}
				// Compute MTTKRP
				val mttkrp = ExtendedBlockMatrix.mttkrp(tensorMatricized(i),
						(for (k <- 0 until result.size if i != k) yield result(k))/*.reverse*/.toArray,
						(for (k <- 0 until tensor.dimensionsSize.size if i != k) yield tensor.dimensionsSize(k)).toArray,
						tensor.dimensionsSize(i),
						rank
					)
				result(i) = mttkrp.multiply(v.pinverse())
				
				// Compute lambda
				lambda = result(i).norm()
				result(i) = result(i).applyOperation(m => {
					for (k <- 0 until rank) {
						m(::,k) := m(::,k) / lambda(k)
					}
					m
				})
				
				// Update of V
				v = v.hadamard(result(i).transpose.multiply(result(i)))
			}
			
			if (nbIterations >= maxIterations) {
				termination = true
			} else {
				nbIterations += 1
			}
		}
		Kruskal(result, lambda)
	}
}
