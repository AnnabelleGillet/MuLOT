package mulot.core

trait Tensor {
	val order: Int
	val dimensionsSize: Array[_ <: AnyVal]
	val dimensionsName: Array[String]
}
