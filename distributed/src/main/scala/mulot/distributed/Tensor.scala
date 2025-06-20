package mulot.distributed

import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRowMatrix, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.connector.expressions.Expressions.literal
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{array, col, row_number, udf}
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}

class Tensor (val data: DataFrame,
			 val order: Int,
			 val dimensionsSize: Array[Long],
			 val dimensionsName: Array[String],
			 val dimensionsIndex: Array[DataFrame],
			 val valueColumnName: String = "val")(implicit spark: SparkSession) extends mulot.core.Tensor {
	/**
	 * Computes the forbenius norm of this [[Tensor]], by adding the absolute value of all the
	 * values of the tensor.
	 */
	def frobeniusNorm(): Double = {
        val _valueColumnName = valueColumnName
		math.sqrt(data.rdd.aggregate(0.0)((v, r) => {
			val currentValue = r.getDouble(r.fieldIndex(_valueColumnName))
			v + (currentValue * currentValue)
		}, _ + _))
	}
	
	/**
	 * Matricizes the tensor on the mode n.
	 *
	 * @param n: the mode of the matricization.
	 * @param transpose: if the matrix should be transposed
	 *
	 * @return IndexedRowMatrix
	 */
	def matricization(n: Int, transpose: Boolean = false): IndexedRowMatrix = {
		def computeIndex(order: Int, currentIndex: Int, dimensionsSize: Array[Long]) = udf((columns: Seq[Long]) => {
			var mul = 1L
			var j = 0L
			for (i <- 0 until order) {
				if (i != currentIndex) {
					j += columns(i) * mul
					mul *= dimensionsSize(i)
				}
			}
			j
		})
		var tensorData = data
		var columnsArr = Array[Column]()
		for (col <- tensorData.columns) {
			if (col.startsWith("row")) {
				columnsArr :+= tensorData.col(col)
			}
		}
		
		tensorData = tensorData.withColumn(s"col", computeIndex(order, n, dimensionsSize)(array(columnsArr: _*)))
		
		import spark.implicits._
		
		val _valueColumnName = valueColumnName
		
		if (transpose) {
			new CoordinateMatrix(
				tensorData.withColumnRenamed(s"row_$n", "row")
					.select("row", "col", _valueColumnName)
					.map(r => MatrixEntry(
							r.getLong(r.fieldIndex("col")),
							r.getLong(r.fieldIndex("row")),
							r.getDouble(r.fieldIndex(_valueColumnName)))
					).rdd,
				(for (i <- dimensionsSize.indices if i != n) yield dimensionsSize(i)).product,
				dimensionsSize(n)
			).toIndexedRowMatrix()
		} else {
			new CoordinateMatrix(
				tensorData.withColumnRenamed(s"row_$n", "row")
					.select("row", "col", _valueColumnName)
					.map(r => 	MatrixEntry(
							r.getLong(r.fieldIndex("row")),
							r.getLong(r.fieldIndex("col")),
							r.getDouble(r.fieldIndex(_valueColumnName)))
					).rdd,
				dimensionsSize(n),
				(for (i <- dimensionsSize.indices if i != n) yield dimensionsSize(i)).product
			).toIndexedRowMatrix()
		}
	}
	
	/**
	 * Matricizes the tensor on the mode n.
	 *
	 * @param n: the mode of the matricization.
	 *
	 * @return List[[RDD[(Int, MatrixEntry)]]]
	 */
	def matricizationToRdd(n: Int): RDD[(Int, MatrixEntry)] = {
		def computeIndex(order: Int, currentIndex: Int, dimensionsSize: Array[Long]) = udf((columns: Seq[Long]) => {
			var mul = 1L
			var j = 0L
			for (i <- 0 until order) {
				if (i != currentIndex) {
					j += columns(i) * mul
					mul *= dimensionsSize(i)
				}
			}
			j
		})
		var tensorData = data
		var columnsArr = Array[Column]()
		for (col <- tensorData.columns) {
			if (col.startsWith("row")) {
				columnsArr :+= tensorData.col(col)
			}
		}
		
		tensorData = tensorData.withColumn(s"col", computeIndex(order, n, dimensionsSize)(array(columnsArr: _*)))
		
		val _valueColumnName = valueColumnName

		tensorData.select(s"row_$n", s"col", _valueColumnName)
			.withColumnRenamed(s"row_$n", "row")
			.rdd.map(r => math.ceil(r.get(r.fieldIndex("row")).toString.toLong / 1024).toInt ->
			MatrixEntry(
				r.get(r.fieldIndex("row")).toString.toLong,
				r.get(r.fieldIndex("col")).toString.toLong,
				r.get(r.fieldIndex(_valueColumnName)).toString.toDouble)
			)
	}
	
	private[mulot] def reindex(dimension: Int, newIndex: DataFrame): Tensor = {
		
		val newData = data.join(dimensionsIndex(dimension), data.col(s"row_$dimension") === dimensionsIndex(dimension).col("dimIndex"))
			.drop(s"row_$dimension")
			.withColumnRenamed("dimIndex", "oldIndex")
			.join(newIndex, "dimValue")
			.drop("dimValue", "oldIndex").withColumnRenamed("dimIndex", s"row_$dimension")
		
		val newIndexes = for (i <- dimensionsIndex.indices) yield {
			if (i == dimension) newIndex else dimensionsIndex(i)
		}
		val newDimensionsSize = for (i <- dimensionsSize.indices) yield {
			if (i == dimension) newIndex.count() else dimensionsSize(i)
		}
		new Tensor(newData, order, newDimensionsSize.toArray, dimensionsName, newIndexes.toArray)
	}
}

object Tensor {
	/**
	 * Creates a tensor from a [[DataFrame]], the columns other than `valueColumnName`
	 * are considered as the dimension's values. The dimensions' values can be of any type,
	 * they will be transformed to [[Long]] in the tensor.
	 *
	 * @return [[Tensor]]
	 */
	def apply(data: DataFrame, valueColumnName: String = "val")(implicit spark: SparkSession): Tensor = {
		var dimensionsIndex = Array.empty[DataFrame]
		var dimensionsSize = Array.empty[Long]
		var dimensionsName = Array.empty[String]
		var tensorData = data
		var i = 0
		for (col <- data.columns) {
			if (col != valueColumnName) {
				val columnIndex = createIndex(data.select(col).distinct())
				dimensionsIndex :+= columnIndex
				dimensionsSize :+= columnIndex.count()
				dimensionsName :+= col
				tensorData = tensorData.join(columnIndex, columnIndex.col("dimValue") === tensorData.col(col))
				tensorData = tensorData.drop("dimValue").withColumnRenamed("dimIndex", s"row_$i").drop(col)
				i += 1
			}
		}
		
		tensorData = tensorData
			.withColumn(valueColumnName, col(valueColumnName).cast(org.apache.spark.sql.types.DoubleType))
		
		new Tensor(tensorData,
			data.columns.length - 1,
			dimensionsSize,
			dimensionsName,
			dimensionsIndex,
			valueColumnName)
	}
	
	/**
	 * Creates a tensor from a [[DataFrame]], the columns other than `valueColumnName`
	 * are considered as the dimension's values. The dimensions' values are the indexes of
	 * the tensor's values.
	 *
	 * @return [[Tensor]]
	 */
	def fromIndexedDataFrame(data: DataFrame, dimensionsSize: Array[Long], valueColumnName: String = "val")(implicit spark: SparkSession): Tensor = {
		var tensorData = data
		var dimensionsName = Array.empty[String]
		var dimensionsIndex = Array.empty[DataFrame]
		var i = 0
		for (columnName <- data.columns) {
			if (columnName != valueColumnName) {
				dimensionsName :+= columnName
				val columnIndex = data.select(columnName).distinct().withColumnRenamed(columnName, "dimValue").withColumn("dimIndex", col("dimValue").cast(org.apache.spark.sql.types.LongType))
				dimensionsIndex :+= columnIndex
				tensorData = tensorData
					.withColumn(columnName, col(columnName).cast(org.apache.spark.sql.types.LongType))
					.withColumnRenamed(columnName, s"row_$i")
				i += 1
			}
		}
		
		tensorData = tensorData
			.withColumn(valueColumnName, col(valueColumnName).cast(org.apache.spark.sql.types.DoubleType))
		
		new Tensor(tensorData,
			data.columns.length - 1,
			dimensionsSize,
			dimensionsName,
			dimensionsIndex,
			valueColumnName)
	}
	
	private def createIndex(df: DataFrame)(implicit spark: SparkSession): DataFrame = {
		spark.sqlContext.createDataFrame(
			df.rdd.zipWithIndex.map {
				case (row, index) => Row.fromSeq(row.toSeq :+ index)
			},
			// Create schema for index column
			StructType(df.withColumnRenamed(df.columns(0), "dimValue").schema.fields :+ StructField("dimIndex", LongType, false)))
	}
	
	private[mulot] def reindexDimension(tensors: Array[(Tensor, Int)]): DataFrame = {
		var newIndex = tensors.head._1.dimensionsIndex(tensors.head._2)
		var currentSize = newIndex.count()
		for ((tensor, dimension) <- tensors.tail) {
			var newElements = tensor.dimensionsIndex(dimension).join(newIndex, "dimValue", "leftanti")
			if (!newElements.head(1).isEmpty) { // Add the elements that are not already in index
				newElements = newElements.withColumn("dimIndex", row_number().over(Window.orderBy("dimIndex")))
				newElements = newElements.withColumn("dimIndex", col("dimIndex") + (currentSize - 1))
				newIndex = newIndex.union(newElements)
				currentSize += newElements.count()
			}
		}
		newIndex
	}
}

