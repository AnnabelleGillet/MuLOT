package mulot.core.tensordecomposition.tucker

import mulot.core.Tensor
import mulot.core.tensordecomposition.{AbstractHOOIResult, Decomposition}

trait HOOI[TensorType <: Tensor, FactorMatricesType, ExplicitValuesType] extends Decomposition[TensorType, FactorMatricesType, ExplicitValuesType] {
	
	type Return <: HOOI[TensorType, FactorMatricesType, ExplicitValuesType]
	type DR = HOOIResult
	
	protected val ranks: Array[Int]
	override protected[mulot] var convergenceThreshold = 10E-5
	private[mulot] var initializer: (TensorType, Array[Int]) => Array[FactorMatricesType]
	protected def resultToExplicitValues(result: HOOIResult): ExplicitValuesType
	
	override protected def copy(): Return = {
		val newDecomposition = super.copy()
		newDecomposition.initializer = this.initializer
		newDecomposition
	}
	
	case class HOOIResult(override val U: Array[FactorMatricesType], override val coreTensor: TensorType) extends AbstractHOOIResult[FactorMatricesType, TensorType] {
		/**
		 * Transform this HOOIResult object to a result with explicit values.
		 */
		def toExplicitValues(): ExplicitValuesType = resultToExplicitValues(this)
	}
	
	/**
	 * Execute the HOOI decomposition with the given parameters.
	 */
	override def execute(): HOOIResult
	
	/**
	 * Choose which method used to initialize factor matrices.
	 *
	 * @param initializer the method to use
	 */
	def withInitializer(initializer: (TensorType, Array[Int]) => Array[FactorMatricesType]): Return = {
		val newObject = this.copy()
		newObject.initializer = initializer
		newObject
	}
}
