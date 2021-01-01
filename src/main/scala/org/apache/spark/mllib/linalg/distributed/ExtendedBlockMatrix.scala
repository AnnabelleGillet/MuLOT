package org.apache.spark.mllib.linalg.distributed

import breeze.linalg.{CSCMatrix, pinv, DenseMatrix => BDM, Matrix => BM}
import org.apache.spark.mllib.linalg._
import org.apache.spark.rdd.{PairRDDFunctions, RDD}
import org.apache.spark.sql.SparkSession

class ExtendedBlockMatrix(blocks: RDD[((Int, Int), Matrix)],
						  rowsPerBlock: Int,
						  colsPerBlock: Int,
						  nRows: Long,
						  nCols: Long) extends BlockMatrix(blocks, rowsPerBlock, colsPerBlock, nRows, nCols) {
	/**
	 * Performs the Hadamard operation between this [[BlockMatrix]] to `other`, another [[BlockMatrix]].
	 * The `rowsPerBlock` of this matrix must equal the `rowsPerBlock` of `other`,
	 * and the `colsPerBlock` of this matrix must equal the `colsPerBlock` of `other`.
	 */
	def hadamard(other: ExtendedBlockMatrix,
				 op: (BM[Double], BM[Double]) => BM[Double] = (m1, m2) => m1 *:* m2): ExtendedBlockMatrix = {
		this.blockMap(other, op)
	}
	
	/**
	 * Applies the `op` globally on this [[ExtendedBlockMatrix]].
	 */
	def applyOperation(op: BDM[Double] => BDM[Double]): ExtendedBlockMatrix = {
		new ExtendedBlockMatrix(
			blocks.map(b => {
				b._2 match {
					case _: DenseMatrix => {
						(b._1, Matrices.fromBreeze(op(b._2.asBreeze.asInstanceOf[BDM[Double]])))
					}
					case _: SparseMatrix => {
						(b._1, Matrices.fromBreeze(op(b._2.asBreeze.asInstanceOf[CSCMatrix[Double]].toDense)))
					}
				}
				
			}).reduceByKey(createPartitioner(), (a, b) => Matrices.fromBreeze(a.asBreeze + b.asBreeze)),
			rowsPerBlock,
			colsPerBlock,
			nRows,
			nCols
		)
	}
	
	/**
	 * Converts this [[ExtendedBlockMatrix]] to a sparse Breeze matrix.
	 */
	def toSparseBreeze(): CSCMatrix[Double] = {
		val m = CSCMatrix.zeros[Double](numRows().toInt, numCols().toInt)
		for (r <- this.toCoordinateMatrix().entries.collect()) {
			val row = r.i.toInt
			val col = r.j.toInt
			val value = r.value
			m(row, col) = value
		}
		m
	}
	
	/**
	 * Computes the norm of each column of this [[ExtendedBlockMatrix]], by adding the absolute value of all the
	 * values of the vector.
	 */
	def norm(): Array[Double] = {
		this.blocks.aggregate((for (i <- 0L until nCols) yield 0.0).toArray)((norm, _m1) => {
			val ((_, _), m1) = _m1
			m1.foreachActive((i, j, v) => norm(j) += Math.abs(v))
			norm
		}, (u1, u2) => (for (i <- u1.indices) yield u1(i) + u2(i)).toArray)
	}
	
	/**
	 * Computes the forbenius norm of this [[ExtendedBlockMatrix]], by adding the absolute value of all the
	 * values of the matrix.
	 */
	def frobeniusNorm(): Double = {
		math.sqrt(this.toCoordinateMatrix().entries.aggregate(0.0)((p, me) => p + math.pow(me.value, 2), (p1, p2) => p1 + p2))
	}
	
	/**
	 * Applies the `pinv` function from Breeze.
	 */
	def pinverse()(implicit spark: SparkSession): ExtendedBlockMatrix = {
		ExtendedBlockMatrix.fromBreeze(pinv(this.toSparseBreeze()))
	}
}

object ExtendedBlockMatrix {
	/**
	 * Converts a [[BlockMatrix]] to a [[ExtendedBlockMatrix]].
	 */
	implicit def fromBlockMatrix(matrix: BlockMatrix): ExtendedBlockMatrix = {
		new ExtendedBlockMatrix(matrix.blocks,
			matrix.rowsPerBlock,
			matrix.colsPerBlock,
			matrix.numRows(),
			matrix.numCols())
	}
	
	/**
	 * Converts a Breeze matrix to a [[ExtendedBlockMatrix]].
	 */
	def fromBreeze(matrix: BM[Double])(implicit spark: SparkSession): ExtendedBlockMatrix = {
		var data = Seq[MatrixEntry]()
		matrix.foreachPair((key, value) => {
			data +:= MatrixEntry(key._1.toLong, key._2.toLong, value)
		})
		
		new CoordinateMatrix(spark.sparkContext.parallelize(data), matrix.rows.toLong, matrix.cols.toLong).toBlockMatrix()
	}
	
	/**
	 * Creates an [[ExtendedBlockMatrix]] with random values.
	 */
	def random(nbRows: Long, nbCols: Long)(implicit spark: SparkSession): ExtendedBlockMatrix = {
		val randomMatrixEntries = spark.sparkContext.parallelize(
			for (i <- 0L until nbRows; j <- 0L until nbCols) yield
				MatrixEntry(i, j, scala.util.Random.nextDouble())
		)
		
		new CoordinateMatrix(randomMatrixEntries, nbRows, nbCols).toBlockMatrix()
	}
	
	/**
	 * Creates an [[ExtendedBlockMatrix]] with gaussian values.
	 */
	def gaussian(nbRows: Long, nbCols: Long)(implicit spark: SparkSession): ExtendedBlockMatrix = {
		val randomMatrixEntries = spark.sparkContext.parallelize(
			for (i <- 0L until nbRows; j <- 0L until nbCols) yield
				MatrixEntry(i, j, scala.util.Random.nextGaussian())
		)
		
		new CoordinateMatrix(randomMatrixEntries, nbRows, nbCols).toBlockMatrix()
	}
	
	/**
	 * Performs the "MTTKRP" (Matricized Tensor Times Khatri Rao Product) between the tensor as
	 * a [[CoordinateMatrix]], and a list of [[BlockMatrix]].
	 *
	 */
	def mttkrp(tensor: PairRDDFunctions[Int, MatrixEntry], dimensions: Array[BlockMatrix], dimensionsSize: Array[Long],
			   currentDimensionSize: Long, rank: Int,
			   rowsPerBlock: Int = 1024, colsPerBlock: Int = 1024): BlockMatrix = {
		require(rowsPerBlock > 0,
			s"rowsPerBlock needs to be greater than 0. rowsPerBlock: $rowsPerBlock")
		require(colsPerBlock > 0,
			s"colsPerBlock needs to be greater than 0. colsPerBlock: $colsPerBlock")
		
		val othersBlocks = (for (matrix <- dimensions) yield matrix.blocks.collect())
		
		val blocks: RDD[((Int, Int), Matrix)] = tensor.aggregateByKey(CSCMatrix.zeros[Double](math.min(rowsPerBlock, currentDimensionSize.toInt), rank))((m, entry) => {
				// Row id in the resulting block
				val rowId = entry.i % rowsPerBlock
				
				// Find the corresponding index in each matrices of the khatri rao product
				var currentJ = entry.j
				val dimensionsIndex = for (i <- dimensionsSize.indices) yield {
					val dimensionJ = (currentJ % dimensionsSize(i)).toInt
					currentJ -= dimensionJ
					currentJ /= dimensionsSize(i)
					dimensionJ
				}
				
				// Find the corresponding value in the corresponding block of the matrices of the khatri rao product
				for (r <- 0 until rank) yield {
					var value = entry.value
					for (i <- dimensionsIndex.indices) {
						val currentIndex = dimensionsIndex(i)
						val currentMatrix = othersBlocks(i).filter { case ((blockRowIndex, blockColIndex), block) => {
							val newIndex = if (currentIndex == 0) 0 else math.ceil(currentIndex / rowsPerBlock).toInt
							newIndex == blockRowIndex
						}}.take(1).head
						value *= currentMatrix._2(currentIndex % rowsPerBlock, r)
					}
					
					m(rowId.toInt, r) += value
				}
				m
			}, _ +:+ _)
			.map { case (blockRowIndex, matrix) => ((blockRowIndex, 0), Matrices.fromBreeze(matrix)) }
		
		new BlockMatrix(blocks, rowsPerBlock, colsPerBlock, currentDimensionSize, rank)
	}
}