package org.apache.spark.mllib.linalg.distributed

import breeze.linalg.{Matrix => BM}
import mulot.distributed.Tensor
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema
import org.apache.spark.sql.{Row, SparkSession}

class ExtendedIndexedRowMatrix(rows: RDD[IndexedRow],
							   private var nRows: Long,
							   private var nCols: Int) extends IndexedRowMatrix(rows, nRows, nCols) {
	/**
	 * Performs the product mode-n between the given tensor and the transpose of the current matrix.
	 *
	 * @param tensor the tensor on which to perform the mode-n product
	 * @param mode   the mode of the product
	 * @return
	 */
	def modeNProductWithTranspose(tensor: Tensor, mode: Int)(implicit spark: SparkSession): Tensor = {
		val _valueColumnName = tensor.valueColumnName
		val order = tensor.order
		
		val newTensorData = tensor.data.rdd.keyBy(r => r.getLong(r.fieldIndex(s"row_$mode")))
			// Group the vectors of the matrix with the entries of the tensor that will be multiplied
			.cogroup(rows.keyBy(v => v.index))
			// Create a new entry for each column of the matrix
			.flatMap(v => {
				val vector = v._2._2.head.vector
				v._2._1.flatMap(row => {
					for ((indice, value) <- vector.activeIterator) yield {
						new GenericRowWithSchema((for (field <- row.schema if field.name.startsWith("row")) yield {
							if (field.name == s"row_$mode") {
								indice.toLong
							} else {
								row.getLong(row.fieldIndex(field.name))
							}
						}).toArray[Any] :+ (value * row.getDouble(row.fieldIndex(_valueColumnName))), row.schema)
					}
				})
			})
			// Group all the entries of the tensor that have the same indexes for all the dimensions
			.keyBy(r => for (i <- 0 until order) yield r.getLong(r.fieldIndex(s"row_$i")))
			// Add the values of the tensor that have been grouped
			.aggregateByKey(0.0)((v, r) => r.getDouble(r.fieldIndex(_valueColumnName)) + v, (v1, v2) => v1 + v2)
			// Produce a RDD of Row
			.map { case (key, value) => {
				Row.fromSeq(key :+ value)
			}
			}
		
		// Convert to a Tensor
		new Tensor(spark.createDataFrame(newTensorData, tensor.data.schema),
			tensor.order,
			(for (i <- tensor.dimensionsSize.indices) yield {
				if (i == mode) nCols else tensor.dimensionsSize(i)
			}).toArray,
			tensor.dimensionsName,
			tensor.dimensionsIndex,
			tensor.valueColumnName
		)
	}
	
	def VofSVD(rank: Int)(implicit spark: SparkSession): ExtendedIndexedRowMatrix = {
		ExtendedIndexedRowMatrix.fromBreeze(this.computeSVD(rank).V.asBreeze)
	}
}

object ExtendedIndexedRowMatrix {
	/**
	 * Converts a [[IndexedRowMatrix]] to a [[ExtendedIndexedRowMatrix]].
	 */
	implicit def fromIndexedRowMatrix(matrix: IndexedRowMatrix): ExtendedIndexedRowMatrix = {
		new ExtendedIndexedRowMatrix(matrix.rows,
			matrix.numRows(),
			matrix.numCols().toInt)
	}
	
	/**
	 * Converts a Breeze matrix to a [[ExtendedIndexedRowMatrix]].
	 */
	def fromBreeze(matrix: BM[Double])(implicit spark: SparkSession): ExtendedIndexedRowMatrix = {
		var data = Seq[MatrixEntry]()
		matrix.foreachPair((key, value) => {
			data +:= MatrixEntry(key._1.toLong, key._2.toLong, value)
		})
		
		new CoordinateMatrix(spark.sparkContext.parallelize(data), matrix.rows.toLong, matrix.cols.toLong).toIndexedRowMatrix()
	}
	
	/**
	 * Creates an [[ExtendedIndexedRowMatrix]] with gaussian values.
	 */
	def gaussian(nbRows: Long, nbCols: Long)(implicit spark: SparkSession): ExtendedIndexedRowMatrix = {
		val randomMatrixEntries = spark.sparkContext.parallelize(
			for (i <- 0L until nbRows; j <- 0L until nbCols) yield
				MatrixEntry(i, j, scala.util.Random.nextGaussian())
		)
		
		new CoordinateMatrix(randomMatrixEntries, nbRows, nbCols).toIndexedRowMatrix()
	}
}
