package mulot.core.tensordecomposition.tucker

import mulot.core.Tensor

trait HOOI[T1 <: Tensor[_], T2, T3] {
	protected val tensor: T1
	protected val ranks: Array[Int]
	protected var maxIterations: Int = 25
	protected var minFrobenius: Double = 10E-5
	protected var initializer: (T1, Array[Int]) => Array[T2]
	
	protected def resultToExplicitValues(result: HOOIResult): T3
	
	private def internalCopy(): HOOI[T1, T2, T3] = {
		val newObject = this.copy()
		newObject.maxIterations = this.maxIterations
		newObject.minFrobenius = this.minFrobenius
		newObject.initializer = this.initializer
		newObject
	}
	
	protected def copy(): HOOI[T1, T2, T3]
	
	case class HOOIResult(U: Array[T2], coreTensor: T1) {
		/**
		 * Transform this HOOIResult object to a result with explicit values.
		 */
		def toExplicitValues(): T3 = resultToExplicitValues(this)
	}
	
	/**
	 * Execute the HOOI decomposition with the given parameters.
	 */
	def execute(): HOOIResult
	
	/**
	 * The maximal number of iterations to perform before stopping the algorithm if the convergence criteria is not met.
	 *
	 * @param maxIterations the number of iterations
	 */
	def withMaxIterations(maxIterations: Int): HOOI[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.maxIterations = maxIterations
		newObject
	}
	
	/**
	 * The Frobenius norm is used as convergence criteria to determine when to stop the iteration.
	 * It represents the similarity between the core tensors of two iterations, with a value between 0 and 1 (at 0
	 * the core tensors are completely different, and they are the same at 1).
	 *
	 * @param minFrobenius the threshold of the Frobenius norm at which stopping the iteration
	 */
	def withMinFrobenius(minFrobenius: Double): HOOI[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.minFrobenius = minFrobenius
		newObject
	}
	
	/**
	 * Choose which method used to initialize factor matrices.
	 *
	 * @param initializer the method to use
	 */
	def withInitializer(initializer: (T1, Array[Int]) => Array[T2]): HOOI[T1, T2, T3] = {
		val newObject = this.internalCopy()
		newObject.initializer = initializer
		newObject
	}
}
