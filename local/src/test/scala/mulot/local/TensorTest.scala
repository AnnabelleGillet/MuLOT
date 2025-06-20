package mulot.local

import breeze.linalg.DenseMatrix
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.must.Matchers.contain
import org.scalatest.matchers.should.Matchers.{convertToAnyShouldWrapper, equal}

class TensorTest extends AnyFunSuite {
	val dataMT = Map[Array[Int], Double](
		Array(0, 0, 0, 0) -> 1.0,
		Array(1, 0, 0, 0) -> 2.0,
		Array(0, 0, 0, 1) -> 3.0,
		Array(1, 0, 0, 1) -> 4.0,
		Array(0, 0, 1, 0) -> 5.0,
		Array(1, 0, 1, 0) -> 6.0,
		Array(0, 0, 1, 1) -> 7.0,
		Array(1, 0, 1, 1) -> 8.0,
		Array(0, 1, 1, 1) -> 9.0,
		Array(1, 1, 1, 1) -> 10.0,
		Array(2, 1, 1, 1) -> 11.0
	)
	val tensor = Tensor.fromIndexedMap(dataMT, 4, Array(3, 2, 2, 2), Array("d0", "d1", "d2", "d3"))
	
	test("test reindexation of tensor") {
		val dataTensor1 = Map[Array[_], Double](
			Array(0, 0) -> 1.0,
			Array(1, 0) -> 2.0,
			Array(2, 0) -> 3.0,
			Array(0, 1) -> 4.0,
			Array(1, 1) -> 5.0,
			Array(2, 1) -> 6.0,
			Array(0, 2) -> 7.0,
			Array(1, 2) -> 8.0,
			Array(2, 2) -> 9.0
		)
		val tensor1 = Tensor(dataTensor1, 2, Array("Dim1", "Dim2"))
		
		val dataTensor2 = Map[Array[Int], Double](
			Array(0, 0) -> 1.0,
			Array(1, 0) -> 2.0,
			Array(2, 0) -> 3.0,
			Array(0, 1) -> 4.0,
			Array(1, 1) -> 5.0,
			Array(2, 1) -> 6.0,
			Array(0, 2) -> 7.0,
			Array(1, 2) -> 8.0,
			Array(2, 2) -> 9.0
		)
		val tensor2 = Tensor.fromIndexedMap(dataTensor2, 2, Array(3, 3), Array("Dim1", "Dim2"))
		
		for (i <- 0 until 2; j <- 0 until 2) {
			val newIndex = Tensor.reindexDimension(Array((tensor1, i), (tensor2, j)))
			val newInverseIndex = newIndex.map(e => e._2 -> e._1)
			val newTensor1 = tensor1.reindex(i, newIndex, newInverseIndex)
			val newTensor2 = tensor2.reindex(j, newIndex, newInverseIndex)
			newTensor1.dimensionsIndex(i) should contain theSameElementsAs newTensor2.dimensionsIndex(j)
			newTensor1.inverseDimensionsIndex(i) should contain theSameElementsAs newTensor2.inverseDimensionsIndex(j)
			val rebuildedTensor1 = for ((keys, value) <- newTensor1.tensorIntegerData) yield {
				val newKey = (for (i <- keys.indices) yield {
					newTensor1.inverseDimensionsIndex(i)(keys(i))
				}).toArray
				
				newKey -> value
			}
			rebuildedTensor1.map(e => e._1.toSeq :+ e._2) should contain theSameElementsAs tensor1.data.map(e => e._1.toSeq :+ e._2)
			val rebuildedTensor2 = for ((keys, value) <- newTensor2.tensorIntegerData) yield {
				val newKey = (for (i <- keys.indices) yield {
					newTensor2.inverseDimensionsIndex(i)(keys(i))
				}).toArray
				
				newKey -> value
			}
			rebuildedTensor2.map(e => e._1.toSeq :+ e._2) should contain theSameElementsAs tensor2.data.map(e => e._1.toSeq :+ e._2)
		}
	}
	
	test("test reindexation of tensor with different sizes of dimensions") {
		val dataTensor1 = Map[Array[_], Double](
			Array(0, 0) -> 1.0,
			Array(1, 0) -> 2.0,
			Array(0, 1) -> 4.0,
			Array(1, 1) -> 5.0
		)
		val tensor1 = Tensor(dataTensor1, 2, Array("Dim1", "Dim2"))
		
		val dataTensor2 = Map[Array[Int], Double](
			Array(0, 0) -> 1.0,
			Array(1, 0) -> 2.0,
			Array(2, 0) -> 3.0,
			Array(0, 1) -> 4.0,
			Array(1, 1) -> 5.0,
			Array(2, 1) -> 6.0,
			Array(0, 2) -> 7.0,
			Array(1, 2) -> 8.0,
			Array(2, 2) -> 9.0
		)
		val tensor2 = Tensor.fromIndexedMap(dataTensor2, 2, Array(3, 3), Array("Dim1", "Dim2"))
		
		for (i <- 0 until 2; j <- 0 until 2) {
			val newIndex = Tensor.reindexDimension(Array((tensor1, i), (tensor2, j)))
			val newInverseIndex = newIndex.map(e => e._2 -> e._1)
			val newTensor1 = tensor1.reindex(i, newIndex, newInverseIndex)
			val newTensor2 = tensor2.reindex(j, newIndex, newInverseIndex)
			for (k <- 0 until 2) {
				if (k != i) {
					newTensor1.dimensionsSize(k) should equal (2)
				}
				if (k != j) {
					newTensor2.dimensionsSize(k) should equal (3)
				}
			}
			newTensor1.dimensionsSize(i) should equal (3)
			newTensor2.dimensionsSize(j) should equal (3)
			newTensor1.dimensionsIndex(i) should contain theSameElementsAs newTensor2.dimensionsIndex(j)
			newTensor1.inverseDimensionsIndex(i) should contain theSameElementsAs newTensor2.inverseDimensionsIndex(j)
			val rebuildedTensor1 = for ((keys, value) <- newTensor1.tensorIntegerData) yield {
				val newKey = (for (i <- keys.indices) yield {
					newTensor1.inverseDimensionsIndex(i)(keys(i))
				}).toArray
				
				newKey -> value
			}
			rebuildedTensor1.map(e => e._1.toSeq :+ e._2) should contain theSameElementsAs tensor1.data.map(e => e._1.toSeq :+ e._2)
			val rebuildedTensor2 = for ((keys, value) <- newTensor2.tensorIntegerData) yield {
				val newKey = (for (i <- keys.indices) yield {
					newTensor2.inverseDimensionsIndex(i)(keys(i))
				}).toArray
				
				newKey -> value
			}
			rebuildedTensor2.map(e => e._1.toSeq :+ e._2) should contain theSameElementsAs tensor2.data.map(e => e._1.toSeq :+ e._2)
		}
	}
	
	test("test matricization mode 0") {
		val dataCompareResult = DenseMatrix.zeros[Double](3, 8)
		dataCompareResult(0, 0) = 1.0
		dataCompareResult(1, 0) = 2.0
		dataCompareResult(0, 4) = 3.0
		dataCompareResult(1, 4) = 4.0
		dataCompareResult(0, 2) = 5.0
		dataCompareResult(1, 2) = 6.0
		dataCompareResult(0, 6) = 7.0
		dataCompareResult(1, 6) = 8.0
		dataCompareResult(0, 7) = 9.0
		dataCompareResult(1, 7) = 10.0
		dataCompareResult(2, 7) = 11.0
		
		val result = tensor.matricization(0)
		
		assert(result.cols == 8)
		assert(result.rows == 3)
		assert(result == dataCompareResult)
	}
	
	test("test matricization mode 1") {
		val dataCompareResult = DenseMatrix.zeros[Double](2, 12)
		dataCompareResult(0, 0) = 1.0
		dataCompareResult(0, 1) = 2.0
		dataCompareResult(0, 6) = 3.0
		dataCompareResult(0, 7) = 4.0
		dataCompareResult(0, 3) = 5.0
		dataCompareResult(0, 4) = 6.0
		dataCompareResult(0, 9) = 7.0
		dataCompareResult(0, 10) = 8.0
		dataCompareResult(1, 9) = 9.0
		dataCompareResult(1, 10) = 10.0
		dataCompareResult(1, 11) = 11.0
		
		val result = tensor.matricization(1)
		
		assert(result.cols == 12)
		assert(result.rows == 2)
		assert(result == dataCompareResult)
	}
	
	test("test matricization mode 2") {
		val dataCompareResult = DenseMatrix.zeros[Double](2, 12)
		dataCompareResult(0, 0) = 1.0
		dataCompareResult(0, 1) = 2.0
		dataCompareResult(0, 6) = 3.0
		dataCompareResult(0, 7) = 4.0
		dataCompareResult(1, 0) = 5.0
		dataCompareResult(1, 1) = 6.0
		dataCompareResult(1, 6) = 7.0
		dataCompareResult(1, 7) = 8.0
		dataCompareResult(1, 9) = 9.0
		dataCompareResult(1, 10) = 10.0
		dataCompareResult(1, 11) = 11.0
		
		val result = tensor.matricization(2)
		
		assert(result.cols == 12)
		assert(result.rows == 2)
		assert(result == dataCompareResult)
	}
	
	test("test chained matricization") {
		val dataCompareResult1 = DenseMatrix.zeros[Double](3, 8)
		dataCompareResult1(0, 0) = 1.0
		dataCompareResult1(1, 0) = 2.0
		dataCompareResult1(0, 4) = 3.0
		dataCompareResult1(1, 4) = 4.0
		dataCompareResult1(0, 2) = 5.0
		dataCompareResult1(1, 2) = 6.0
		dataCompareResult1(0, 6) = 7.0
		dataCompareResult1(1, 6) = 8.0
		dataCompareResult1(0, 7) = 9.0
		dataCompareResult1(1, 7) = 10.0
		dataCompareResult1(2, 7) = 11.0
		
		val result1 = tensor.matricization(0)
		
		assert(result1.cols == 8)
		assert(result1.rows == 3)
		assert(result1 == dataCompareResult1)
		
		val dataCompareResult2 = DenseMatrix.zeros[Double](2, 12)
		dataCompareResult2(0, 0) = 1.0
		dataCompareResult2(0, 1) = 2.0
		dataCompareResult2(0, 6) = 3.0
		dataCompareResult2(0, 7) = 4.0
		dataCompareResult2(0, 3) = 5.0
		dataCompareResult2(0, 4) = 6.0
		dataCompareResult2(0, 9) = 7.0
		dataCompareResult2(0, 10) = 8.0
		dataCompareResult2(1, 9) = 9.0
		dataCompareResult2(1, 10) = 10.0
		dataCompareResult2(1, 11) = 11.0
		
		val result2 = tensor.matricization(1)
		
		assert(result2.cols == 12)
		assert(result2.rows == 2)
		assert(result2 == dataCompareResult2)
		
		val dataCompareResult3 = DenseMatrix.zeros[Double](2, 12)
		dataCompareResult3(0, 0) = 1.0
		dataCompareResult3(0, 1) = 2.0
		dataCompareResult3(0, 6) = 3.0
		dataCompareResult3(0, 7) = 4.0
		dataCompareResult3(1, 0) = 5.0
		dataCompareResult3(1, 1) = 6.0
		dataCompareResult3(1, 6) = 7.0
		dataCompareResult3(1, 7) = 8.0
		dataCompareResult3(1, 9) = 9.0
		dataCompareResult3(1, 10) = 10.0
		dataCompareResult3(1, 11) = 11.0
		
		val result3 = tensor.matricization(2)
		
		assert(result3.cols == 12)
		assert(result3.rows == 2)
		assert(result3 == dataCompareResult3)
	}
	
	test("test transposed matricization mode 0") {
		val dataCompareResult = DenseMatrix.zeros[Double](8, 3)
		dataCompareResult(0, 0) = 1.0
		dataCompareResult(0, 1) = 2.0
		dataCompareResult(4, 0) = 3.0
		dataCompareResult(4, 1) = 4.0
		dataCompareResult(2, 0) = 5.0
		dataCompareResult(2, 1) = 6.0
		dataCompareResult(6, 0) = 7.0
		dataCompareResult(6, 1) = 8.0
		dataCompareResult(7, 0) = 9.0
		dataCompareResult(7, 1) = 10.0
		dataCompareResult(7, 2) = 11.0
		
		val result = tensor.matricization(0, true)
		
		assert(result.cols == 3)
		assert(result.rows == 8)
		assert(result == dataCompareResult)
	}
	
	test("test transposed matricization mode 1") {
		val dataCompareResult = DenseMatrix.zeros[Double](12, 2)
		dataCompareResult(0, 0) = 1.0
		dataCompareResult(1, 0) = 2.0
		dataCompareResult(6, 0) = 3.0
		dataCompareResult(7, 0) = 4.0
		dataCompareResult(3, 0) = 5.0
		dataCompareResult(4, 0) = 6.0
		dataCompareResult(9, 0) = 7.0
		dataCompareResult(10, 0) = 8.0
		dataCompareResult(9, 1) = 9.0
		dataCompareResult(10, 1) = 10.0
		dataCompareResult(11, 1) = 11.0
		
		val result = tensor.matricization(1, true)
		
		assert(result.cols == 2)
		assert(result.rows == 12)
		assert(result == dataCompareResult)
	}
	
	test("test transposed matricization mode 2") {
		val dataCompareResult = DenseMatrix.zeros[Double](12, 2)
		dataCompareResult(0, 0) = 1.0
		dataCompareResult(1, 0) = 2.0
		dataCompareResult(6, 0) = 3.0
		dataCompareResult(7, 0) = 4.0
		dataCompareResult(0, 1) = 5.0
		dataCompareResult(1, 1) = 6.0
		dataCompareResult(6, 1) = 7.0
		dataCompareResult(7, 1) = 8.0
		dataCompareResult(9, 1) = 9.0
		dataCompareResult(10, 1) = 10.0
		dataCompareResult(11, 1) = 11.0
		
		val result = tensor.matricization(2, true)
		
		assert(result.cols == 2)
		assert(result.rows == 12)
		assert(result == dataCompareResult)
	}
}
