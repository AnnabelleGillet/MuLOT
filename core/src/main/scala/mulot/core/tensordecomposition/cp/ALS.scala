package mulot.core.tensordecomposition.cp

import mulot.core.Tensor

trait ALS[T1 <: Tensor[_], T2, T3] {
	protected val tensor: T1
	protected val rank: Int
	protected var maxIterations: Int = 25
	protected var minFms: Double = 0.99
	protected var computeCorcondia: Boolean = false
	protected var norm: Norms.Norm = Norms.L2
	protected var initializer: (T1, Int) => Array[T2]
	
	protected def kruskalToExplicitValues(kruskal: Kruskal): T3
	private def internalCopy(): ALS[T1, T2, T3] = {
		val newObject = this.copy()
		newObject.maxIterations = this.maxIterations
		newObject.minFms = this.minFms
		newObject.computeCorcondia = this.computeCorcondia
		newObject.norm = this.norm
		newObject.initializer = this.initializer
		newObject
	}
	protected def copy(): ALS[T1, T2, T3]
	
	case class Kruskal(A: Array[T2], lambdas: Array[Double], corcondia: Option[Double]) {
		/**
		 * Transform this Kruskal object to a result with explicit values.
		 */
		def toExplicitValues(): T3 = kruskalToExplicitValues(this)
	}
	
	/**
	 * Execute the CP decomposition with the given parameters.
	 */
	def execute(): Kruskal
	
	/**
	 * The norm to use to normalize the factor matrices.
	 *
	 * @param norm the chosen norm
	 */
	def withNorm(norm: Norms.Norm): ALS[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.norm = norm
		newObject
	}
	
	/**
	 * The maximal number of iterations to perform before stopping the algorithm if the convergence criteria is not met.
	 *
	 * @param maxIterations the number of iterations
	 */
	def withMaxIterations(maxIterations: Int): ALS[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.maxIterations = maxIterations
		newObject
	}
	
	/**
	 * The Factor Match Score (FMS) is used as convergence criteria to determine when to stop the iteration.
	 * It represents the similarity between the factor matrices of two iterations, with a value between 0 and 1 (at 0
	 * the matrices are completely different, and they are the same at 1).
	 *
	 * @param minFms the threshold of the FMS at which stopping the iteration
	 */
	def withMinFms(minFms: Double): ALS[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.minFms = minFms
		newObject
	}
	
	/**
	 * Choose if CORCONDIA must be computed after the execution of the decomposition.
	 *
	 * @param computeCorcondia true to compute CORCONDIA, false otherwise
	 */
	def withComputeCorcondia(computeCorcondia: Boolean): ALS[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.computeCorcondia = computeCorcondia
		newObject
	}
	
	/**
	 * Choose which method used to initialize factor matrices.
	 *
	 * @param initializer the method to use
	 */
	def withInitializer(initializer: (T1, Int) => Array[T2]): ALS[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.initializer = initializer
		newObject
	}
}

object Norms extends Enumeration {
	type Norm = Value
	
	val L1, L2 = Value
}
