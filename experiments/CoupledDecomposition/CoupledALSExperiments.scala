import breeze.linalg._

import mulot.local.Tensor
import mulot.local.tensordecomposition._
import mulot.local.tensordecomposition.cp.ALS
import mulot.local.tensordecomposition.cp.ALS._
import mulot.local.tensordecomposition.cp.CoupledALS
import mulot.local.tensordecomposition.cp.CoupledALS._
import mulot.local.tensordecomposition.cp.CoupledCP
import mulot.core.tensordecomposition.CoupledDimension
import mulot.core.tensordecomposition.cp._

import java.awt.Color
import collection.JavaConverters._

import smile.data._
import smile.data.`type`._
import smile.plot.swing._

object CoupledALSExperimentsV2 {
	val groundTruth = (for (i <- 0 until 30) yield {
		if (i < 10) 0
		else if (i < 15) 1
		else if (i < 20) 2
		else 3
	}).toArray
		
	/**
	 * Create a dense area over the given values of range of dimension. 
	 */
	def createCluster(dimension1: Range, dimension2: Range, dimension3: Range, value: Double = 10.0): Map[Array[Int], Double] = {
		val rand = new scala.util.Random
		(for (i <- dimension1; j <- dimension2; k <- dimension3) yield {
		    Array(i, j, k) -> (value + (rand.nextInt(6) - 3))
		}).toMap
	}
	
	/**
	 * Add some noise in data.
	 */
	def createNoise(nb: Long, dimension1: Range, _possibilities: Array[(Int, Int)], value: Double = 10.0): Map[Array[Int], Double] = {
		val rand = new scala.util.Random
		var data = Map.empty[Array[Int], Double]
		for (dim1 <- dimension1) {
			var possibilities = (for (p <- _possibilities) yield p).toArray
			var _nb = nb
			while (_nb > 0 && possibilities.nonEmpty) {
				val p = rand.nextInt(possibilities.size)
				val possibility = possibilities(p)
				possibilities = (for (j <- possibilities.indices if j != p) yield possibilities(j)).toArray
				data += Array(dim1, possibility._1, possibility._2) -> (value + (rand.nextInt(6) - 3))
				_nb -= 1
			}
		}
		data
	}

	// Execute with "scala -classpath lib/*:. CoupledALSExperiments.scala"
	// Scala version used: 2.12.20
	def main(args: Array[String]): Unit = {
		// Build main tensor: 3 clusters that span over 10 elements of each dimension.
		var mainTensorData = Map[Array[Int], Double]()
		mainTensorData ++= createCluster(0 until 10, 0 until 10, 0 until 10)
		mainTensorData ++= createCluster(10 until 20, 10 until 20, 10 until 20)
		mainTensorData ++= createCluster(20 until 30, 20 until 30, 20 until 30)
		val mainTensor = Tensor.fromIndexedMap(mainTensorData, 3, Array(30, 30, 30), Array("dimension1", "dimension2", "dimension3"))
		
		// Build second tensor: 2 clusters that span over 15 elements of the first dimension and 5 elements of the second and third dimensions.
		var secondTensorData = Map[Array[Int], Double]()
		secondTensorData ++= createCluster(0 until 15, 0 until 5, 0 until 5)
		secondTensorData ++= createCluster(15 until 30, 5 until 10, 5 until 10)
		val secondTensor = Tensor.fromIndexedMap(secondTensorData, 3, Array(30, 10, 10), Array("dimension1", "dimension2", "dimension3"))

		baselineDecomposition(mainTensor, 4)
		simpleExperiment(mainTensor, secondTensor, 4)
		
		for (n <- List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)) {
			missingDataExperiment(mainTensorData, secondTensor, 4, n / 100)
		}

		for (n <- List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)) {
			noisyDataExperiment(mainTensorData, secondTensorData, 4, n / 100)
		}
	}
	
	/**
	 * Execute a CP decomposition on the main tensor.
	 */
	def baselineDecomposition(mainTensor: Tensor, rank: Int): Unit = {
		// Baseline decomposition
		val baselineDecomposition = ALS(mainTensor, rank).withMaxIterations(500).withPrintEvery(100)
		val resultBaselineDecomposition = baselineDecomposition.execute()

		// Clusterize
		val clusters = clusterize(resultBaselineDecomposition.A(0))
		println(s"Index for baseline experiment:")
		evaluateClusters(clusters, resultBaselineDecomposition.A(0))

		// Produce vizualisation of result
		plotMatrix(resultBaselineDecomposition.A(0), "Baseline")
	}
	
	/**
	 * Execute a coupled CP decomposition on the main tensor and the second tensor. 
	 */
	def simpleExperiment(mainTensor: Tensor, secondTensor: Tensor, rank: Int): Unit = {
		// Coupled decomposition
		val decomposition1 = ALS(mainTensor, 3).withMaxIterations(500).withPrintEvery(100)//.withInitializer(ALS.Initializers.hosvd)
		val decomposition2 = ALS(secondTensor, 2).withMaxIterations(500).withPrintEvery(100)//.withInitializer(ALS.Initializers.hosvd)
		val coupledDecomposition = new CoupledCP(Array(decomposition1, decomposition2), Array(0, 0)).withThreshold(0.6)
		val resultCoupledDecomposition = coupledDecomposition.execute()

		// Clusterize
		val clusters = clusterize(resultCoupledDecomposition.A(0)(0))
		println(s"Index for simple experiment (proposed algorithm):")
		evaluateClusters(clusters, resultCoupledDecomposition.A(0)(0))

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecomposition.A(0)(0), "Simple experiment (proposed algorithm)")
		
		// De Lathauwer
		val coupledDecompositionDL = CoupledALS(Array(mainTensor, secondTensor), rank, Array(CoupledDimension(mainTensor, secondTensor, Map(0 -> 0)))).withMaxIterations(500).withPrintEvery(100)//.withInitializer(CoupledALS.Initializers.hosvd)
		val resultCoupledDecompositionDL = coupledDecompositionDL.execute()
		
		// Clusterize
		val clustersDL = clusterize(resultCoupledDecompositionDL.A(0)(0))
		println(s"Index for simple experiment (De Lathauwer algorithm):")
		evaluateClusters(clustersDL, resultCoupledDecompositionDL.A(0)(0))

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecompositionDL.A(0)(0), "Simple experiment (De Lathauwer algorithm)")
	}
	
	/**
	 * Remove elements of clusters only in one tensor to see how the coupled decomposition deals with missing data.
	 */
	def missingDataExperiment(mainTensorData: Map[Array[Int], Double], secondTensor: Tensor, rank: Int, missingDataPercent: Double): Unit = {
		var removedKeys = (createNoise((10 * 10 * missingDataPercent).toInt, 0 until 10, (for (d2 <- 0 until 10; d3 <- 0 until 10) yield (d2, d3)).toArray) ++ createNoise((10 * 10 * missingDataPercent).toInt, 10 until 20, (for (d2 <- 10 until 20; d3 <- 10 until 20) yield (d2, d3)).toArray) ++ createNoise((10 * 10 * missingDataPercent).toInt, 20 until 30, (for (d2 <- 20 until 30; d3 <- 20 until 30) yield (d2, d3)).toArray)).keys.toArray
		
		val newMainTensorData = mainTensorData.filterKeys(k => !removedKeys.exists(k2 => k2.sameElements(k)))
		val newMainTensor = Tensor.fromIndexedMap(newMainTensorData, 3, Array(30, 30, 30), Array("dimension1", "dimension2", "dimension3"))
		
		// Coupled decomposition
		val decomposition1 = ALS(newMainTensor, 3).withMaxIterations(500).withPrintEvery(100)//.withInitializer(ALS.Initializers.hosvd)
		val decomposition2 = ALS(secondTensor, 2).withMaxIterations(500).withPrintEvery(100)//.withInitializer(ALS.Initializers.hosvd)
		val coupledDecomposition = new CoupledCP(Array(decomposition1, decomposition2), Array(0, 0)).withThreshold(0.6)
		val resultCoupledDecomposition = coupledDecomposition.execute()
		
		// Clusterize
		val clusters = clusterize(resultCoupledDecomposition.A(0)(0))
		println(s"Index for ${missingDataPercent * 100}% missing data experiment (proposed algorithm):")
		evaluateClusters(clusters, resultCoupledDecomposition.A(0)(0))

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecomposition.A(0)(0), s"Missing data experiment with ${missingDataPercent * 100}% of missing data (proposed algorithm)")
		
		// De Lathauwer
		val coupledDecompositionDL = CoupledALS(Array(newMainTensor, secondTensor), rank, Array(CoupledDimension(newMainTensor, secondTensor, Map(0 -> 0)))).withMaxIterations(500).withPrintEvery(100)//.withInitializer(CoupledALS.Initializers.hosvd)
		val resultCoupledDecompositionDL = coupledDecompositionDL.execute()
		
		// Clusterize
		val clustersDL = clusterize(resultCoupledDecompositionDL.A(0)(0))
		println(s"Index for ${missingDataPercent * 100}% missing data experiment (De Lathauwer algorithm):")
		evaluateClusters(clustersDL, resultCoupledDecompositionDL.A(0)(0))

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecompositionDL.A(0)(0), s"Missing data experiment with ${missingDataPercent * 100}% of missing data (De Lathauwer algorithm)") 
	}
	 
	/**
	 * Add noise in the main tensor to see how the decomposition performs. 
	 */
	def noisyDataExperiment(mainTensorData: Map[Array[Int], Double], secondTensorData: Map[Array[Int], Double], rank: Int, noisePercent: Double): Unit = {
		var mainAddedEntries = createNoise((((30 * 30) - (10 * 10)) * noisePercent).toInt, 0 until 10, (for (d2 <- 0 until 30; d3 <- 0 until 30 if d2 >= 10 || d3 >= 10) yield (d2, d3)).toArray)
		mainAddedEntries ++= createNoise((((30 * 30) - (10 * 10)) * noisePercent).toInt, 10 until 20, (for (d2 <- 0 until 30; d3 <- 0 until 30 if !(d2 >= 10 && d2 < 20) || !(d3 >= 10 && d3 < 20)) yield (d2, d3)).toArray)
		mainAddedEntries ++= createNoise((((30 * 30) - (10 * 10)) * noisePercent).toInt, 20 until 30, (for (d2 <- 0 until 30; d3 <- 0 until 30 if d2 <= 20 || d3 <= 20) yield (d2, d3)).toArray)
		val newMainTensorData = mainTensorData ++ mainAddedEntries
		val newMainTensor = Tensor.fromIndexedMap(newMainTensorData, 3, Array(30, 30, 30), Array("dimension1", "dimension2", "dimension3"))
		
		var secondAddedEntries = createNoise((((10 * 10) - (5 * 5)) * noisePercent).toInt, 0 until 15, (for (d2 <- 0 until 10; d3 <- 0 until 10 if d2 >= 5 || d3 >= 5) yield (d2, d3)).toArray)
		secondAddedEntries ++= createNoise((((10 * 10) - (5 * 5)) * noisePercent).toInt, 15 until 30, (for (d2 <- 0 until 10; d3 <- 0 until 10 if d2 <= 5 || d3 <= 5) yield (d2, d3)).toArray)
		val newSecondTensorData = secondTensorData ++ secondAddedEntries
		val newSecondTensor = Tensor.fromIndexedMap(newSecondTensorData, 3, Array(30, 10, 10), Array("dimension1", "dimension2", "dimension3"))
		
		// Coupled decomposition
		val decomposition1 = ALS(newMainTensor, 3).withMaxIterations(500).withPrintEvery(100)//.withInitializer(ALS.Initializers.hosvd)
		val decomposition2 = ALS(newSecondTensor, 2).withMaxIterations(500).withPrintEvery(100)//.withInitializer(ALS.Initializers.hosvd)
		val coupledDecomposition = new CoupledCP(Array(decomposition1, decomposition2), Array(0, 0)).withThreshold(0.6)
		val resultCoupledDecomposition = coupledDecomposition.execute()
		
		// Clusterize
		val clusters = clusterize(resultCoupledDecomposition.A(0)(0))
		println(s"Index for noisy data experiment with ${noisePercent * 100}% of noise (proposed algorithm):")
		evaluateClusters(clusters, resultCoupledDecomposition.A(0)(0))

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecomposition.A(0)(0), s"Noisy data experiment with ${noisePercent * 100}% of noise (proposed algorithm)")
		
		// De Lathauwer
		val coupledDecompositionDL = CoupledALS(Array(newMainTensor, newSecondTensor), rank, Array(CoupledDimension(newMainTensor, newSecondTensor, Map(0 -> 0)))).withMaxIterations(500).withPrintEvery(100)//.withInitializer(CoupledALS.Initializers.hosvd)
		val resultCoupledDecompositionDL = coupledDecompositionDL.execute()
		
		// Clusterize
		val clustersDL = clusterize(resultCoupledDecompositionDL.A(0)(0))
		println(s"Index for ${noisePercent * 100}% noisy data experiment (De Lathauwer algorithm):")
		evaluateClusters(clustersDL, resultCoupledDecompositionDL.A(0)(0))

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecompositionDL.A(0)(0), s"Noisy data experiment with ${noisePercent * 100}% of noise (De Lathauwer algorithm)") 
	}
	 
	 
	 
	// Data structure for vizualisation
	val struct = DataTypes.struct(new StructField("factor1", DataTypes.DoubleType), new StructField("factor2", DataTypes.DoubleType), new StructField("cluster", DataTypes.StringType))
	case class Entry(factor1: Double, factor2: Double, cluster: String) extends smile.data.Tuple {
		override def schema(): StructType = struct
		override def get(x: Int): Object = if (x == 0) factor1.asInstanceOf[Object] else if (x == 1) factor2.asInstanceOf[Object] else if (x == 2) cluster.asInstanceOf[Object] else null
	}
	 
	/**
	 * Produce a vizualisation for the given matrix.
	 */
	def plotMatrix(matrix: DenseMatrix[Double], title: String = ""): Unit = {
		heatmapPlot(matrix, title)
	}
	 
	def tSnePlot(matrix: DenseMatrix[Double], title: String = ""): Unit = {
		val grid = new PlotGrid(matrix.cols, matrix.cols)
		for (f1 <- 0 until matrix.cols; f2 <- 0 until matrix.cols) {
			val content = (for (i <- 0 until 30) yield {
				Entry(
						matrix(i, f1),
						matrix(i, f2),
						if (i < 10) "C1" else if (i < 15) "C2" else if (i < 20) "C3" else "C4"
				)
			}).toList
			
			var df: DataFrame = DataFrame.of(content.asJava, struct)
			val canvas = ScatterPlot.of(df, "factor1", "factor2", "cluster", '*').canvas()
			canvas.extendLowerBound(Array(-.2, -.2))
			canvas.extendUpperBound(Array(0.6, 0.6))
			canvas.setTitle(title)
			canvas.setAxisLabels(s"Factor $f1", s"Factor $f2")
			grid.add(canvas.panel)
		}
		grid.window()
	}
	 
	def heatmapPlot(matrix: DenseMatrix[Double], title: String = ""): Unit = {
		val n = 32
		val palette = new Array[Color](n)
		for (i <- 0 until (n / 2)) {
			palette(i) = new Color((n - (i * 2)).toFloat / n.toFloat, 0.0f, 0.0f, 0.8f)
			palette(n - i - 1) = new Color(0.0f, (n - (1.0f + i * 2)).toFloat / n.toFloat, (n - (1.0f + i * 2)).toFloat / n.toFloat, 0.8f)
		}
		val xLabel = for (i <- 0 until matrix.cols) yield s"${i + 1}"
		val yLabel = for (i <- 0 until matrix.rows) yield s"${i + 1}"
		val matrixMax = max(matrix)
		val matrixMin = min(matrix)
		val div = if (matrixMax > -matrixMin) matrixMax else -matrixMin
		val _matrix = matrix.map(_ / div)
		val canvas = new Heatmap(yLabel.toArray, xLabel.toArray, _matrix.t.toArray.grouped(matrix.cols).toArray, palette.reverse).canvas()
		canvas.setAxisLabels(s"Factors", s"Elements")
		canvas.setTitle(title)
		canvas.window()
	}
	 
	def clusterize(matrix: DenseMatrix[Double]): List[List[(Int, Double)]] = {
		val vectors = (for (i <- 0 until matrix.cols) yield matrix(::, i)).toList
		Clustering(vectors).run()
	}
	
	def evaluateClusters(clusters: List[List[(Int, Double)]], matrix: DenseMatrix[Double]) = {
		println(s"Rand Index:${Clustering.computeRandIndex(clusters, groundTruth)}")
		val data = clusters.map(cluster => {
			cluster.map(e => (e._1, matrix(e._1, ::).t))
		})
		println(s"Davies Bouldin:${Clustering.computeDaviesBouldinIndex(data)}")
	}
}

/**
 * Clustering method.
 */
class Clustering private(data: List[DenseVector[Double]]) {
	def run(): List[List[(Int, Double)]] = {
		data.indices.map(i => {
			var threshold = Double.MaxValue
			val sortedData = data(i).keySet.map(j => (j, data(i)(j))).toList.sortWith((e1, e2) => e1._2 > e2._2)
			sortedData.indices.takeWhile(j => {
				if (j > 0) {
					val e1 = sortedData(j - 1)._2
					val e2 = sortedData(j)._2
					val oldThreshold = threshold
					threshold = e1 - e2
					(e1 - e2) <= (breeze.linalg.sum(data(i)) / (data(i).length * 2))
				} else {
					true
				}
			}).map(sortedData(_)).toList
		}).toList
	}
}

case class DaviesBouldinIndex(index: Double, clustersInnerDistance: Array[Double]) {
	override def toString(): String = {
		s"""
		Index: $index, Clusters similarity:
		${clustersInnerDistance.mkString(" ")}
		"""
	}
}

case class RandIndex(index: Double, adjusted: Double) {
	override def toString(): String = {
		s"""
		Rand Index: $index, Adjusted Rand Index: $adjusted
		"""
	}
}

object Clustering {
	def apply(data: List[DenseVector[Double]]): Clustering = {
		new Clustering(data)
	}
	
	def computeDaviesBouldinIndex(clusters: List[List[(Int, DenseVector[Double])]]): DaviesBouldinIndex = {
		val centroids = for (cluster <- clusters) yield {
			cluster.map(_._2).reduce(_ + _) / cluster.size.toDouble 
		}
		
		val clustersInnerDistance = for (i <- clusters.indices) yield {
			val cluster = clusters(i)
			cluster.map(v => {
				val distance = v._2 - centroids(i)
				distance.t * distance
			}).reduce(_ + _) / cluster.size 
		}
		
		var index = 0.0
		for (i <- clusters.indices) {
			val centroid1 = centroids(i)
			val innerDistance1 = clustersInnerDistance(i)
			var max = 0.0
			for (j <- clusters.indices if i != j) {
				val centroid2 = centroids(j)
				val innerDistance2 = clustersInnerDistance(j)
				val differenceCentroids = centroid1 - centroid2
				val distanceCentroids =  differenceCentroids.t * differenceCentroids
				val r = (innerDistance1 + innerDistance2) / distanceCentroids
				if (r > max) {
					max = r
				}
			}
			index += max
		}
		index /= clusters.size
		
		DaviesBouldinIndex(index, clustersInnerDistance.toArray)
	}
	
	def computeRandIndex(_clusters: List[List[(Int, Double)]], groundTruth: Array[Int]): RandIndex = {
		val confusionMatrix = Array.ofDim[Int](_clusters.size, groundTruth.max + 1)
		val clusters = Array.fill[Int](groundTruth.size)(-1) 
		for (i <- _clusters.indices) {
			for (v <- _clusters(i)) {
				if (clusters(v._1) >= 0) {
					if (_clusters(clusters(v._1)).filter(_._1 == v._1).head._2 < v._2) {
						clusters(v._1) = i
					}
				} else {
					clusters(v._1) = i
				}
			}
		}
		for (i <- groundTruth.indices) {
			if (clusters(i) != -1) {
				confusionMatrix(clusters(i))(groundTruth(i)) += 1
			}
		}
		
		// Grouped in clusters and grouped in ground truth
		var a = 0
		// Grouped in clusters but separated in ground truth
		var b = 0
		// Separated in clusters but grouped in groud truth
		var c = 0
		// Separated in clusters and separated in ground truth
		var d = 0
		
		for (i1 <- clusters.indices; i2 <- clusters.indices if i2 > i1 && clusters(i1) >= 0 && clusters(i2) >= 0) {
			if (clusters(i1) == clusters(i2)) {
				if (groundTruth(i1) == groundTruth(i2)) {
					a += 1
				} else {
					b += 1
				}
			} else {
				if (groundTruth(i1) == groundTruth(i2)) {
					c += 1
				} else {
					d += 1
				}
			}
		}
		
		def factorial(n: Int): BigInt = n match {
			case 0 => 1
			case _ => (BigInt(1) to BigInt(n)).reduce(_ * _)
		}
		
		def combination(n: Int, r: Int): Double = {
			if (n < r) 0.0 else factorial(n).toDouble / (factorial(r) * factorial(n - r)).toDouble
		}
		
		var s = 0.0
		for (i1 <- _clusters.indices; i2 <- confusionMatrix(i1).indices) {
			s += combination(confusionMatrix(i1)(i2), 2)
		}
		var s1 = 0.0
		var s2 = 0.0
		for (i <- _clusters.indices) {
			s1 += combination(clusters.filter(_ == i).size, 2)
		}
		for (i <- confusionMatrix(0).indices) {
			s2 += combination(groundTruth.filter(_ == i).size, 2)
		}
		val ari = (s - (s1 * s2 / combination(clusters.size, 2))) / (((s1 + s2) / 2) - (s1 * s2 / combination(clusters.size, 2)))
		
		RandIndex((a + d).toDouble / (a + b + c + d + clusters.filter(_ == -1).size).toDouble, ari)
	}
}
