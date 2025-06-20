package mulot.core.tensordecomposition

import mulot.core.Tensor

trait Decomposition[TensorType <: Tensor, FactorMatricesType, ExplicitValuesType] {
	
	private[mulot] type Return <: Decomposition[TensorType, FactorMatricesType, ExplicitValuesType]
	private[mulot] type DR <: DecompositionResult[FactorMatricesType]
	
	private[mulot] var tensor: TensorType
	
	private[mulot] var maxIterations: Int = 25
	
	private[mulot] var computeConvergence: Boolean = true
	
	private[mulot] var convergenceThreshold: Double
	
	private[core] var convergenceMethod: (DR, DR) => Double
	
	private[core] def execute(): DR
	
	protected def copy(): Return = {
		val newDecomposition = internalCopy()
		newDecomposition.maxIterations = this.maxIterations
		newDecomposition.computeConvergence = this.computeConvergence
		newDecomposition.convergenceThreshold = this.convergenceThreshold
		newDecomposition.convergenceMethod = this.convergenceMethod.asInstanceOf[(newDecomposition.DR, newDecomposition.DR) => Double]
		newDecomposition
	}
	protected def internalCopy(): Return
	
	/**
	 * The maximal number of iterations to perform before stopping the algorithm if the convergence criteria is not met.
	 *
	 * @param maxIterations the number of iterations
	 */
	def withMaxIterations(maxIterations: Int): Return = {
		val newObject = this.copy()
		newObject.maxIterations = maxIterations
		newObject
	}
	
	/**
	 * Choose if the convergence must be computed at each iteration.
	 *
	 * @param computeConvergence true if the convergence is computed, false otherwise.
	 */
	def withComputeConvergence(computeConvergence: Boolean): Return = {
		val newDecomposition = copy()
		newDecomposition.computeConvergence = computeConvergence
		newDecomposition
	}
	
	/**
	 * Set the convergence threshold to use to check convergence of the algorithm. The result of the convergence method
	 * is compared against this convergence threshold: if it is lower the algorithm stops, otherwise it continues.
	 *
	 * @param convergenceThreshold the convergence threshold to use.
	 */
	def withConvergenceThreshold(convergenceThreshold: Double): Return = {
		val newDecomposition = copy()
		newDecomposition.convergenceThreshold = convergenceThreshold
		newDecomposition
	}
	
	/**
	 * Set the convergence method to use to check convergence of the algorithm. The result of this method is compared
	 * against the convergence threshold: if it is lower the algorithm stops, otherwise it continues.
	 *
	 * @param convergenceMethod the convergence method to use.
	 */
	def withConvergenceMethod(convergenceMethod: (DR, DR) => Double): Return = {
		val newDecomposition = copy()
		newDecomposition.convergenceMethod = convergenceMethod.asInstanceOf[(newDecomposition.DR, newDecomposition.DR) => Double]
		newDecomposition
	}
}
