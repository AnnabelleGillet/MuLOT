package mulot.distributed.tensordecomposition.tucker

import mulot.core.tensordecomposition.tucker
import mulot.distributed.Tensor
import org.apache.spark.mllib.linalg.distributed.ExtendedIndexedRowMatrix
import org.apache.spark.sql.{DataFrame, SparkSession}
import scribe.Logging

object HOOI {
	def apply(tensor: Tensor, ranks: Array[Int])(implicit spark: SparkSession): HOOI = {
		var columnsName = (for (i <- 0 until tensor.order) yield s"row_$i") :+ tensor.valueColumnName
		var newTensor = new Tensor(
			tensor.data.select(columnsName(0), columnsName.tail: _*).cache(),
			tensor.order,
			tensor.dimensionsSize,
			tensor.dimensionsName,
			tensor.dimensionsIndex,
			tensor.valueColumnName
		)
		new HOOI(newTensor, ranks)(spark)
	}
	
	object Initializers {
		def gaussian(tensor: Tensor, ranks: Array[Int])(implicit spark: SparkSession): Array[ExtendedIndexedRowMatrix] = {
			(for (i <- 0 until tensor.order) yield {
				ExtendedIndexedRowMatrix.gaussian(tensor.dimensionsSize(i), ranks(i))
			}).toArray
		}
		
		def hosvd(tensor: Tensor, ranks: Array[Int])(implicit spark: SparkSession): Array[ExtendedIndexedRowMatrix] = {
			(for (i <- 0 until tensor.order) yield {
				ExtendedIndexedRowMatrix.fromIndexedRowMatrix(tensor.matricization(i, true)).VofSVD(ranks(i))
			}).toArray
		}
	}
}

class HOOI private[tucker](val tensor: Tensor, val ranks: Array[Int])(implicit spark: SparkSession)
	extends tucker.HOOI[Tensor, ExtendedIndexedRowMatrix, Map[String, DataFrame]]
		with Logging {
	
	override var initializer: (Tensor, Array[Int]) => Array[ExtendedIndexedRowMatrix] = HOOI.Initializers.hosvd
	
	override protected def copy(): HOOI = {
		val newObject = new HOOI(tensor, ranks)
		newObject
	}
	
	override protected def resultToExplicitValues(result: HOOIResult): Map[String, DataFrame] = {
		(for (i <- tensor.dimensionsName.indices) yield {
			var df = spark.createDataFrame(result.U(i).toCoordinateMatrix().entries).toDF("dimIndex", "rank", "val")
			if (tensor.dimensionsIndex.isDefined) {
				df = df.join(tensor.dimensionsIndex.get(i), "dimIndex").select("dimValue", "rank", "val")
				df = df.withColumnRenamed("dimValue", tensor.dimensionsName(i))
			} else {
				df = df.withColumnRenamed("dimIndex", tensor.dimensionsName(i))
			}
			tensor.dimensionsName(i) -> df
		}).toMap
	}
	
	override def execute(): HOOIResult = {
		// Initialisation
		val begin = System.currentTimeMillis()
		val factorMatrices = initializer(tensor, ranks)
		println(s"Initialisation in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		
		// Order the dimensions of the tensor to start from the biggest one,
		// so we can reduce quickly the total size of the tensor
		val dimensionsOrder = tensor.dimensionsSize.zipWithIndex.sortWith((v1, v2) => v1._1 >= v2._1).map(v => v._2)
		var convergence = false
		var finalCoreTensor: Tensor = null
		var lastIterationFrobeniusDifference = Double.MaxValue
		val originalFrobenius = tensor.frobeniusNorm()
		var iteration = 0
		
		// Iterate while the convergence criteria is not met
		while (!convergence && iteration <= maxIterations) {
			println(s"Iteration $iteration")
			val tuckerBegin = System.currentTimeMillis()
			var previousCoreTensor = new Tensor(
				tensor.data.cache(),
				tensor.order,
				tensor.dimensionsSize,
				tensor.dimensionsName,
				tensor.dimensionsIndex,
				tensor.valueColumnName
			)
			// Compute the new factor matrices
			for (dimensionIndice <- dimensionsOrder.indices) {
				val dimension = dimensionsOrder(dimensionIndice)
				
				// Prepare the core tensor for the iteration
				var coreTensor = new Tensor(
					previousCoreTensor.data.cache(),
					previousCoreTensor.order,
					previousCoreTensor.dimensionsSize,
					previousCoreTensor.dimensionsName,
					previousCoreTensor.dimensionsIndex,
					previousCoreTensor.valueColumnName
				)
				// Compute the core tensor with mode-n product except with the factor matrix of the current dimension
				for (i <- (dimensionIndice + 1) until tensor.order) {
					val currentDimension = dimensionsOrder(i)
					coreTensor = factorMatrices(currentDimension).modeNProductWithTranspose(coreTensor, currentDimension)
				}
				
				// Compute the new factor matrix for the current dimension
				factorMatrices(dimension) = ExtendedIndexedRowMatrix.fromIndexedRowMatrix(coreTensor.matricization(dimension, true)).VofSVD(ranks(dimension))
				
				// Update the global core tensor
				previousCoreTensor = factorMatrices(dimension).modeNProductWithTranspose(previousCoreTensor, dimension)
			}
			
			// Compute the Frobenius norm of the difference of the current core tensor and of the core tensor
			// of the previous iteration
			val frobenius = previousCoreTensor.frobeniusNorm()
			val residualNorm = math.sqrt(originalFrobenius * originalFrobenius - frobenius * frobenius)
			val frobeniusDifference = 1 - ({
				if (residualNorm.isNaN) 0.0 else residualNorm
			} / originalFrobenius)
			val tolerance = math.abs(lastIterationFrobeniusDifference - frobeniusDifference)
			println(s"Convergence value of the iteration: $tolerance")
			// Check if the difference between the two Frobenius norm is smaller
			// than the convergence criteria
			if (tolerance <= minFrobenius) {
				convergence = true
			}
			lastIterationFrobeniusDifference = frobeniusDifference
			
			// Keep the final core tensor il the convergence criteria is met
			if (convergence || iteration >= maxIterations) {
				finalCoreTensor = previousCoreTensor
			}
			
			iteration += 1
			println(s"Tucker iteration in ${(System.currentTimeMillis() - tuckerBegin).toDouble / 1000.0}s")
		}
		if (finalCoreTensor == null) {
			finalCoreTensor = new Tensor(
				tensor.data.cache(),
				tensor.order,
				tensor.dimensionsSize,
				tensor.dimensionsName,
				tensor.dimensionsIndex,
				tensor.valueColumnName
			)
			for (dimensionIndice <- dimensionsOrder.indices) {
				val dimension = dimensionsOrder(dimensionIndice)
				finalCoreTensor = factorMatrices(dimension).modeNProductWithTranspose(finalCoreTensor, dimension)
			}
		}
		
		var finalData = finalCoreTensor.data
		for (dimension <- 0 until finalCoreTensor.order) {
			finalData = finalData.withColumnRenamed(s"row_$dimension", finalCoreTensor.dimensionsName(dimension))
		}
		finalCoreTensor = new Tensor(
			finalData,
			tensor.order,
			tensor.dimensionsSize,
			tensor.dimensionsName,
			tensor.dimensionsIndex,
			tensor.valueColumnName
		)
		
		HOOIResult(factorMatrices, finalCoreTensor)
	}
}
