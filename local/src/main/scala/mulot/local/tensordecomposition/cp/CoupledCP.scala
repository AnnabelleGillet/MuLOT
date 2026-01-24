package mulot.local.tensordecomposition.cp

import breeze.linalg.{DenseMatrix, DenseVector, inv, max, min}
import breeze.numerics.abs
import mulot.core.tensordecomposition.{AbstractKruskal, CoupledDimension}
import mulot.core.tensordecomposition.cp.Norms
import mulot.local.Tensor
import scribe.Logging

object CoupledCP extends Logging {
	def apply(decompositions: Array[ALS], commonDimensions: Array[Int]): CoupledCP = {
		val newIndex = Tensor.reindexDimension((for (d <- decompositions.indices) yield (decompositions(d).tensor, commonDimensions(d))).toArray)
		val newInverseIndex = newIndex.map(e => e._2 -> e._1)
		val newDecompositions: Array[ALS] = (for (i <- decompositions.indices) yield {
			var newTensor = decompositions(i).tensor
			newTensor = newTensor.reindex(commonDimensions(i), newIndex, newInverseIndex)
			val newDecomposition = decompositions(i).copy()
			newDecomposition.tensor = newTensor
			newDecomposition
		}).toArray
		new CoupledCP(newDecompositions, commonDimensions)
	}
}

class CoupledCP(val decompositions: Array[ALS], commonDimensions: Array[Int]) extends mulot.core.tensordecomposition.cp.ALS[Tensor, Array[DenseMatrix[Double]], Array[Map[String, Array[Map[Any, Double]]]]]
	with Logging {
	
	override type Return = CoupledCP
	override protected val rank: Int = 0
	protected var threshold: Double = 0.5
	override private[mulot] var tensor: Tensor = null
	override private[mulot] var convergenceMethod: (Kruskal, Kruskal, Boolean) => Double = null
	
	def withThreshold(threshold: Double): Return = {
		val newObject = this.copy()
		newObject.threshold = threshold
		newObject
	}
	
	override protected def internalCopy(): Return = {
		val newDecomposition = new CoupledCP(decompositions, commonDimensions)
		newDecomposition.threshold = this.threshold
		newDecomposition
	}
	
	override private[mulot] def copy(): Return = {
		val newDecomposition = super.copy()
		newDecomposition
	}
	
	override protected def kruskalToExplicitValues(kruskal: Kruskal): Array[Map[String, Array[Map[Any, Double]]]] = {
		(for (i <- decompositions.indices) yield {
			val tensor = decompositions(i).tensor
			(for (j <- tensor.dimensionsName.indices) yield {
				val matrix = kruskal.A(i)(j)
				val mapRanks = (for (r <- 0 until matrix.cols) yield {
					(for ((key, value) <- matrix(::, r).iterator) yield {
						tensor.inverseDimensionsIndex(j)(key) -> value
					}).toMap
				}).toArray
				tensor.dimensionsName(j) -> mapRanks
			}).toMap
		}).toArray
	}
	
	override def execute(): Kruskal = {
		val decompositionResults = for (decomposition <- decompositions) yield decomposition.execute()
		val newFactorVectors = for (decomposition <- decompositions) yield (for (_ <- 0 until decomposition.tensor.order) yield List[DenseVector[Double]]()).toArray
		var newLambdas = List[Double]()
		// Define good vectors
		val commonDimension1 = commonDimensions.head
		val matrix1 = decompositionResults(0).factorMatrices(commonDimension1)
		var vectorsList = (for (r1 <- 0 until matrix1.cols) yield {
			val dimensionsIndices = Array.ofDim[Int](decompositions.length)
			dimensionsIndices(0) = r1
			(dimensionsIndices, matrix1(::, r1))
		}).toList
		for (decomposition2Index <- commonDimensions.indices if decomposition2Index > 0) {
			val commonDimension2 = commonDimensions(decomposition2Index)
			val matrix2 = decompositionResults(decomposition2Index).factorMatrices(commonDimension2)
			
			var newVectorsList = List[(Array[Int], DenseVector[Double])]()
			for (r2 <- 0 until matrix2.cols) {
				for ((dimensionsIndices, vector) <- vectorsList) {
					val newVector = vector *:* matrix2(::, r2)
					
					val newDimensionsIndices = dimensionsIndices.clone()
					newDimensionsIndices(decomposition2Index) = r2
					newVectorsList +:= (newDimensionsIndices, newVector)
				}
			}
			vectorsList = newVectorsList
		}
		vectorsList = vectorsList.filter(e => {
			val normVector = breeze.linalg.norm(e._2)
			val distances = for (i <- e._1.indices) yield {
				val v = decompositionResults(i).factorMatrices(commonDimensions(i))(::, e._1(i))
				max(v)
			}
			(max(e._2) / distances.product) > threshold 
		})
		
		// Add remaining vectors to result
		for ((dimensionsIndices, vector) <- vectorsList) {
			newLambdas :+= (for (i <- dimensionsIndices.indices) yield decompositionResults(i).lambdas(dimensionsIndices(i))).product
			for (decompositionIndex <- decompositions.indices) {
				for (o <- 0 until decompositions(decompositionIndex).tensor.order) {
					if (o == commonDimensions(decompositionIndex)) {
						newFactorVectors(decompositionIndex)(o) :+= vector
					} else {
						newFactorVectors(decompositionIndex)(o) :+= decompositionResults(decompositionIndex).factorMatrices(o)(::, dimensionsIndices(decompositionIndex))
					}
				}
			}
		}
		
		// Change list of vectors to DenseMatrix
		val newFactorMatrices = for (result <- newFactorVectors) yield {
			for (vectors <- result) yield {
				val matrix = DenseMatrix.zeros[Double](vectors.head.length, vectors.length)
				for (i <- vectors.indices) {
					val vector = vectors(i)
					matrix(::, i) := vector
				}
				matrix
			}
		}
		
		Kruskal(newFactorMatrices, newLambdas.toArray, None)
	}
}
