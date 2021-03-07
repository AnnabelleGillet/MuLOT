package tensordecomposition

import org.apache.spark.mllib.linalg.distributed.MatrixEntry
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{array, col, udf}
import org.apache.spark.sql.types.{LongType, StructField, StructType}
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}

class Tensor(val data: DataFrame,
			 val order: Int,
			 val dimensionsSize: List[Long],
			 val dimensionsName: List[String],
			 val dimensionsIndex: Option[List[DataFrame]],
			 val valueColumnName: String = "val")(implicit spark: SparkSession) {
	/**
	 * Computes the forbenius norm of this [[Tensor]], by adding the absolute value of all the
	 * values of the tensor.
	 */
	def frobeniusNorm(): Double = {
		math.sqrt(data.rdd.aggregate(0.0)((v, r) => {
			val currentValue = r.getDouble(r.fieldIndex(valueColumnName))
			v + (currentValue * currentValue)
		}, _ + _))
	}
	
	/**
	 * Matricizes the tensor for all the modes.
	 *
	 * @return List[[RDD[(Int, MatrixEntry)]]]
	 */
	def matricization(rowsPerBlock: Int = 1024): List[RDD[(Int, MatrixEntry)]] = {
		def computeIndex(order: Int, currentIndex: Int, dimensionsSize: List[Long]) = udf((columns: Seq[Long]) => {
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
		
		for (i <- 0 until order) {
			tensorData = tensorData.withColumn(s"col_$i", computeIndex(order, i, dimensionsSize)(array(columnsArr: _*)))
		}
		
		var matrices = List[RDD[(Int, MatrixEntry)]]()
		val _valueColumnName = valueColumnName
		for (i <- 0 until order) {
			matrices :+= tensorData.select(s"row_$i", s"col_$i", _valueColumnName)
				.withColumnRenamed(s"col_$i", "col")
				.withColumnRenamed(s"row_$i", "row")
				.rdd.map(r => (math.ceil(r.get(r.fieldIndex("row")).toString.toLong / rowsPerBlock).toInt ->
					MatrixEntry(
						r.get(r.fieldIndex("row")).toString.toLong,
						r.get(r.fieldIndex("col")).toString.toLong,
						r.get(r.fieldIndex(_valueColumnName)).toString.toDouble))
				)
		}
		matrices
	}
	
	/**
	 * Run the CP decomposition for this tensor.
	 *
	 * @param rank
	 * @param nbIterations
	 * @param norm the norm to use on the columns of the factor matrices
	 * @param minFms the Factor Match Score limit to stop the algorithm
	 * @param checkpoint to use or not the checkpoint of Spark (can improve performance for a high number of iterations)
	 * @param highRank improve the computation of the pinverse if set to true. By default, is true when rank >= 100.
	 * @return A [[Map]], with the [[String]] name of each dimension of the tensor mapped to a [[DataFrame]].
	 *         This [[DataFrame]] has 3 columns: one with the values of the original dimension, one with the values of the rank,
	 *         and the last one with the values found with the CP.
	 */
	def runCPALS(rank: Int, nbIterations: Int = 25, norm: String = CPALS.NORM_L1, minFms: Double = 0.99,
				 checkpoint: Boolean = false, highRank: Option[Boolean] = None): Map[String, DataFrame] = {
		val kruskal = CPALS.computeSparkCPALS(this, rank, norm, nbIterations, minFms, checkpoint, highRank)
		
		(for (i <- dimensionsName.indices) yield {
			var df = spark.createDataFrame(kruskal.A(i).toCoordinateMatrixWithZeros().entries).toDF("dimIndex", "rank", "val")
			if (dimensionsIndex.isDefined) {
				df = df.join(dimensionsIndex.get(i), "dimIndex").select("dimValue", "rank", "val")
				df = df.withColumnRenamed("dimValue", dimensionsName(i))
			} else {
				df = df.withColumnRenamed("dimIndex", dimensionsName(i))
			}
			dimensionsName(i) -> df
		}).toMap
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
		var dimensionsIndex = List[DataFrame]()
		var dimensionsSize = List[Long]()
		var dimensionsName = List[String]()
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
		
		new Tensor(tensorData,
			data.columns.size - 1,
			dimensionsSize,
			dimensionsName,
			Some(dimensionsIndex),
			valueColumnName)
	}
	
	/**
	 * Creates a tensor from a [[DataFrame]], the columns other than `valueColumnName`
	 * are considered as the dimension's values. The dimensions' values are the indexes of
	 * the tensor's values.
	 *
	 * @return [[Tensor]]
	 */
	def fromIndexedDataFrame(data: DataFrame, dimensionsSize: List[Long], valueColumnName: String = "val")(implicit spark: SparkSession): Tensor = {
		var tensorData = data
		var dimensionsName = List[String]()
		var i = 0
		for (columnName <- data.columns) {
			if (columnName != valueColumnName) {
				dimensionsName :+= columnName
				tensorData = tensorData
					.withColumn(columnName, col(columnName).cast(org.apache.spark.sql.types.LongType))
					.withColumnRenamed(columnName, s"row_$i")
				i += 1
			}
		}
		
		tensorData = tensorData
			.withColumn(valueColumnName, col(valueColumnName).cast(org.apache.spark.sql.types.DoubleType))
		
		new Tensor(tensorData,
			data.columns.size - 1,
			dimensionsSize,
			dimensionsName,
			None,
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
}

