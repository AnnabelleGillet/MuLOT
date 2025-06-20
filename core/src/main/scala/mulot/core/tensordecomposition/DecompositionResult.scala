package mulot.core.tensordecomposition

trait DecompositionResult[T1] {
	val factorMatrices: Array[T1]
}

trait AbstractKruskal[FactorMatricesType] extends DecompositionResult[FactorMatricesType] {
	val A: Array[FactorMatricesType]
	val lambdas: Array[Double]
	val corcondia: Option[Double]
	
	override val factorMatrices: Array[FactorMatricesType] = A
}

trait AbstractHOOIResult[FactorMatricesType, TensorType] extends DecompositionResult[FactorMatricesType] {
	val U: Array[FactorMatricesType]
	val coreTensor: TensorType
	
	override val factorMatrices: Array[FactorMatricesType] = U
}
