package mulot.core.tensordecomposition.cp

import mulot.core.Tensor
import mulot.core.tensordecomposition.{AbstractKruskal, Decomposition}

trait ALS[TensorType <: Tensor, FactorMatricesType, ExplicitValuesType] extends Decomposition[TensorType, FactorMatricesType, ExplicitValuesType] {
	
	type Return <: ALS[TensorType, FactorMatricesType, ExplicitValuesType]
	type DR = Kruskal
	
	protected val rank: Int
	override private[mulot] var convergenceThreshold: Double = 0.01
	private[mulot] var computeCorcondia: Boolean = false
	private[mulot] var norm: Norms.Norm = Norms.L2
	private[mulot] var nonNegativity: Boolean = false
	
	protected def kruskalToExplicitValues(kruskal: Kruskal): ExplicitValuesType
	
	override protected def copy(): Return = {
		val newDecomposition = super.copy()
		newDecomposition.computeCorcondia = this.computeCorcondia
		newDecomposition.norm = this.norm
		newDecomposition.nonNegativity = this.nonNegativity
		newDecomposition
	}
	
	case class Kruskal(override val A: Array[FactorMatricesType], override val lambdas: Array[Double], override val corcondia: Option[Double]) extends AbstractKruskal[FactorMatricesType] {
		/**
		 * Transform this Kruskal object to a result with explicit values.
		 */
		def toExplicitValues(): ExplicitValuesType = kruskalToExplicitValues(this)
	}
	
	/**
	 * Execute the CP decomposition with the given parameters.
	 */
	override def execute(): Kruskal
	
	/**
	 * The norm to use to normalize the factor matrices.
	 *
	 * @param norm the chosen norm
	 */
	def withNorm(norm: Norms.Norm): Return = {
		val newObject = this.copy()
		newObject.norm = norm
		newObject
	}
	
	/**
	 * Set if the decomposition must produce a non-negative result or not.
	 *
	 * @param nonNegativity true for non-negative decomposition, false otherwise.
	 */
	def withNonNegativity(nonNegativity: Boolean): Return = {
		val newObject = this.copy()
		newObject.nonNegativity = nonNegativity
		newObject
	}
	
	/**
	 * Choose if CORCONDIA must be computed after the execution of the decomposition.
	 *
	 * @param computeCorcondia true to compute CORCONDIA, false otherwise
	 */
	def withComputeCorcondia(computeCorcondia: Boolean): Return = {
		val newObject = this.copy()
		newObject.computeCorcondia = computeCorcondia
		newObject
	}
}

object Norms extends Enumeration {
	type Norm = Value
	
	val L1, L2 = Value
}
