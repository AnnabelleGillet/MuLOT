package mulot.local

import breeze.linalg.DenseMatrix

/**
 * A tensor constructed from a [[Map]], with an [[Array]] of any type as key and [[Double]] as value.
 *
 * @param data the content of the tensor
 * @param order the number of dimensions of the tensor (all the keys of the [[Map]] data must fit this size)
 * @param dimensionsName the name of the dimensions, in the same order as given in the keys
 */
class Tensor private(val data: Map[Array[_], Double],
					 val order: Int,
					 val dimensionsSize: Array[Int],
					 val dimensionsName: Array[String],
					 val dimensionsIndex: Array[Map[Any, Int]],
					 val inverseDimensionsIndex: Array[Map[Int, Any]],
					 val tensorIntegerData: Map[Array[Int], Double]
			) extends mulot.core.Tensor[Map[Array[_], Double]] {
	/**
	 * Matricize the tensor on the mode n.
	 *
	 * @param n the mode of the matricization.
	 * @param transpose if the matrix should be transposed
	 *
	 * @return DenseMatrix
	 */
	def matricization(n: Int, transpose: Boolean = false): DenseMatrix[Double] = {
		val matrix = if (transpose) {DenseMatrix.zeros[Double](dimensionsSize.product / dimensionsSize(n), dimensionsSize(n))}
				else {DenseMatrix.zeros[Double](dimensionsSize(n), dimensionsSize.product / dimensionsSize(n))}
		for ((k, v) <- tensorIntegerData) {
			var j = 0
			var coef = 1
			for (i <- 0 until order if i != n) {
				j += k(i) * coef
				coef *= dimensionsSize(i)
			}
			if (transpose) {
				matrix(j, k(n)) = v
			} else {
				matrix(k(n), j) = v
			}
		}
		matrix
	}
}

object Tensor {
	/**
	 * Create a tensor from a [[Map]], with an [[Array]] of any type as key and [[Double]] as value.
	 *
	 * @param data the content of the tensor
	 * @param order the number of dimensions of the tensor (all the keys of the [[Map]] data must fit this size)
	 * @param dimensionsName the name of the dimensions, in the same order as given in the keys
	 */
	def apply(data: Map[Array[_], Double], order: Int, dimensionsName: Array[String]): Tensor = {
		val dimensionsIndex: Array[Map[Any, Int]] = (for (_ <- 0 until order) yield {Map[Any, Int]()}).toArray
		val inverseDimensionsIndex: Array[Map[Int, Any]] = (for (_ <- 0 until order) yield {Map[Int, Any]()}).toArray
		val tensorIntegerData: Map[Array[Int], Double] = {
			var tmpTensorIntegerData = Map[Array[Int], Double]()
			val currentIndexDimensions: Array[Int] = (for (_ <- 0 until order) yield 0).toArray
			
			for ((keys, value) <- data) yield {
				require(keys.length == order, s"$keys is not of size $order")
				val newKey = new Array[Int](keys.length)
				for (i <- keys.indices) {
					val currentKey = dimensionsIndex(i).get(keys(i))
					if (currentKey.isEmpty) {
						newKey(i) = currentIndexDimensions(i)
						dimensionsIndex(i) += keys(i) -> currentIndexDimensions(i)
						inverseDimensionsIndex(i) += currentIndexDimensions(i) -> keys(i)
						currentIndexDimensions(i) += 1
					} else {
						newKey(i) = currentKey.get
					}
				}
				tmpTensorIntegerData += newKey -> value
			}
			
			tmpTensorIntegerData
		}
		
		val dimensionsSize: Array[Int] = for (dimension <- dimensionsIndex) yield dimension.size
		
		new Tensor(data, order, dimensionsSize, dimensionsName, dimensionsIndex, inverseDimensionsIndex, tensorIntegerData)
	}
	
	/**
	 * Create a tensor from a [[Map]], with an [[Array]] of [[Int]] as key and [[Double]] as value.
	 *
	 * @param data the content of the tensor
	 * @param order the number of dimensions of the tensor
	 * @param dimensionsSize the size of the dimensions, in the same order as given in the keys
	 * @param dimensionsName the name of the dimensions, in the same order as given in the keys
	 */
	def fromIndexedMap(data: Map[Array[Int], Double], order: Int, dimensionsSize: Array[Int], dimensionsName: Array[String]): Tensor = {
		val dimensionsIndex: Array[Map[Any, Int]] = (for (_ <- 0 until order) yield {Map[Any, Int]()}).toArray
		val inverseDimensionsIndex: Array[Map[Int, Any]] = (for (_ <- 0 until order) yield {Map[Int, Any]()}).toArray
		for (i <- 0 until order) {
			for (n <- 0 until dimensionsSize(i)) {
				dimensionsIndex(i) += n -> n
				inverseDimensionsIndex(i) += n -> n
			}
		}
		new Tensor(data.asInstanceOf[Map[Array[_], Double]], order, dimensionsSize, dimensionsName, dimensionsIndex, inverseDimensionsIndex, data)
	}
}
