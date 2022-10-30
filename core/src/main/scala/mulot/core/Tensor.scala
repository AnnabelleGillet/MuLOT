package mulot.core

trait Tensor[T] {
	val data: T
	val order: Int
	val dimensionsSize: Array[_ <: AnyVal]
	val dimensionsName: Array[String]
}
