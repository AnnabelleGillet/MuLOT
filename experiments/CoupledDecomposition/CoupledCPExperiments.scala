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

object CoupledCPExperiments {
	val groundTruth = (for (i <- 0 until 30) yield {
		if (i < 10) 0
		else if (i < 15) 1
		else if (i < 20) 2
		else 3
	}).toArray
	
	var dbiProposedAlgorithm = List[String]()
	var ariProposedAlgorithm = List[String]()
	var dbiDLAlgorithm = List[String]()
	var ariDLAlgorithm = List[String]()
	
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

	// Execute with "scala -classpath lib/*:. CoupledCPExperiments.scala"
	// export LANGUAGE=en:el for nice number formating
	// Scala version used: 3.3.3
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
		
		def resetMeasures() = {
			dbiProposedAlgorithm = List[String]()
			ariProposedAlgorithm = List[String]()
			dbiDLAlgorithm = List[String]()
			ariDLAlgorithm = List[String]()
		}
		
		def printMeasures() = {
			println(s"ARI proposed algorithm: ${ariProposedAlgorithm.mkString(" & ")} \\\\")
			println(s"DBI proposed algorithm: ${dbiProposedAlgorithm.mkString(" & ")} \\\\")
			
			println(s"ARI DL algorithm: ${ariDLAlgorithm.mkString(" & ")} \\\\")
			println(s"DBI DL algorithm: ${dbiDLAlgorithm.mkString(" & ")} \\\\")
		}

		baselineDecomposition(mainTensor, 4)
		simpleExperiment(mainTensor, secondTensor, 4)
		printMeasures()
		
		// Missing data
		resetMeasures()
		for (n <- List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)) {
			missingDataExperiment(mainTensorData, secondTensorData, 4, n / 100)
		}
		printMeasures()
		
		// Noisy data
		resetMeasures()
		for (n <- List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)) {
			noisyDataExperiment(mainTensorData, secondTensorData, 4, n / 100)
		}
		printMeasures()
		
		// Different tensor sizes
		resetMeasures()
		for (n <- List(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)) {
			differentTensorsSizeExperiment(mainTensor, n, 4)
		}
		for (n <- List(100, 500, 1000)) {
			differentTensorsSizeExperiment(mainTensor, n, 4, false)
		}
		printMeasures()
		
		// Different values
		resetMeasures()
		for (n <- List(10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000)) {
			differentValuesExperiment(mainTensor, n, 4)
		}
		printMeasures()
	}
	
	/**
	 * Execute a CP decomposition on the main tensor.
	 */
	def baselineDecomposition(mainTensor: Tensor, rank: Int): Unit = {
		// Baseline decomposition
		val baselineDecomposition = ALS(mainTensor, rank).withMaxIterations(500).withPrintEvery(100).withInitializer(ALS.Initializers.uniform)
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
		runProposedAlgorithm(mainTensor, secondTensor, "simple", "")
		runDLAlgorithm(mainTensor, secondTensor, "simple", "")
	}
	
	/**
	 * Remove elements of clusters in both tensors to see how the coupled decomposition deals with missing data.
	 */
	def missingDataExperiment(mainTensorData: Map[Array[Int], Double], secondTensorData: Map[Array[Int], Double], rank: Int, missingDataPercent: Double): Unit = {
		var removedKeys = (createNoise((10 * 10 * missingDataPercent).toInt, 0 until 10, (for (d2 <- 0 until 10; d3 <- 0 until 10) yield (d2, d3)).toArray) ++ createNoise((10 * 10 * missingDataPercent).toInt, 10 until 20, (for (d2 <- 10 until 20; d3 <- 10 until 20) yield (d2, d3)).toArray) ++ createNoise((10 * 10 * missingDataPercent).toInt, 20 until 30, (for (d2 <- 20 until 30; d3 <- 20 until 30) yield (d2, d3)).toArray)).keys.toArray
		
		val noisePercent = 0.1
		var addedEntries = createNoise((((30 * 30) - (10 * 10)) * noisePercent).toInt, 0 until 10, (for (d2 <- 0 until 30; d3 <- 0 until 30 if d2 >= 10 || d3 >= 10) yield (d2, d3)).toArray)
		addedEntries ++= createNoise((((30 * 30) - (10 * 10)) * noisePercent).toInt, 10 until 20, (for (d2 <- 0 until 30; d3 <- 0 until 30 if !(d2 >= 10 && d2 < 20) || !(d3 >= 10 && d3 < 20)) yield (d2, d3)).toArray)
		addedEntries ++= createNoise((((30 * 30) - (10 * 10)) * noisePercent).toInt, 20 until 30, (for (d2 <- 0 until 30; d3 <- 0 until 30 if d2 <= 20 || d3 <= 20) yield (d2, d3)).toArray)
		
		val newMainTensorData = mainTensorData.filterKeys(k => !removedKeys.exists(k2 => k2.sameElements(k))).toMap ++ addedEntries
		val newMainTensor = Tensor.fromIndexedMap(newMainTensorData, 3, Array(30, 30, 30), Array("dimension1", "dimension2", "dimension3"))
		
		removedKeys = (createNoise((5 * 5 * missingDataPercent).toInt, 0 until 15, (for (d2 <- 0 until 5; d3 <- 0 until 5) yield (d2, d3)).toArray) ++ createNoise((5 * 5 * missingDataPercent).toInt, 15 until 30, (for (d2 <- 5 until 10; d3 <- 5 until 10) yield (d2, d3)).toArray)).keys.toArray
		addedEntries = createNoise((((10 * 10) - (5 * 5)) * noisePercent).toInt, 0 until 15, (for (d2 <- 0 until 10; d3 <- 0 until 10 if d2 >= 5 || d3 >= 5) yield (d2, d3)).toArray)
		addedEntries ++= createNoise((((10 * 10) - (5 * 5)) * noisePercent).toInt, 15 until 30, (for (d2 <- 0 until 10; d3 <- 0 until 10 if d2 <= 5 || d3 <= 5) yield (d2, d3)).toArray)
		
		val newSecondTensorData = secondTensorData.filterKeys(k => !removedKeys.exists(k2 => k2.sameElements(k))).toMap ++ addedEntries
		val newSecondTensor = Tensor.fromIndexedMap(newSecondTensorData, 3, Array(30, 10, 10), Array("dimension1", "dimension2", "dimension3"))
		
		runProposedAlgorithm(newMainTensor, newSecondTensor, "missing data", s"${missingDataPercent * 100}% of missing data")
		runDLAlgorithm(newMainTensor, newSecondTensor, "missing data", s"${missingDataPercent * 100}% of missing data")
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
		
		runProposedAlgorithm(newMainTensor, newSecondTensor, "noisy data", s"${noisePercent * 100}% of noise")
		runDLAlgorithm(newMainTensor, newSecondTensor, "noisy data", s"${noisePercent * 100}% of noise")
	}
	
	/**
	 * Change the size of the non-common dimensions of the second tensor to see how the decomposition performs. 
	 */
	def differentTensorsSizeExperiment(mainTensor: Tensor, dimensionsSize: Int, rank: Int, withDL: Boolean = true): Unit = {
		var secondTensorData = Map[Array[Int], Double]()
		val dimensionSplit = dimensionsSize / 2
		secondTensorData ++= createCluster(0 until 15, 0 until dimensionSplit, 0 until dimensionSplit, 10)
		secondTensorData ++= createCluster(15 until 30, dimensionSplit until dimensionsSize, dimensionSplit until dimensionsSize, 10)
		val secondTensor = Tensor.fromIndexedMap(secondTensorData, 3, Array(30, dimensionsSize, dimensionsSize), Array("dimension1", "dimension2", "dimension3"))
		
		runProposedAlgorithm(mainTensor, secondTensor, "different tensors' size", s"dimensions of size ${dimensionsSize}")
		if (withDL) {
			runDLAlgorithm(mainTensor, secondTensor, "different tensors' size", s"dimensions of size ${dimensionsSize}")
		}
	}
	
	/**
	 * Change the values of the second tensor to see how the decomposition performs. 
	 */
	def differentValuesExperiment(mainTensor: Tensor, values: Int, rank: Int, withDL: Boolean = true): Unit = {
		var secondTensorData = Map[Array[Int], Double]()
		secondTensorData ++= createCluster(0 until 15, 0 until 5, 0 until 5, values)
		secondTensorData ++= createCluster(15 until 30, 5 until 10, 5 until 10, values)
		val secondTensor = Tensor.fromIndexedMap(secondTensorData, 3, Array(30, 10, 10), Array("dimension1", "dimension2", "dimension3"))
		
		runProposedAlgorithm(mainTensor, secondTensor, "different values", s"values of ${values}")
		if (withDL) {
			runDLAlgorithm(mainTensor, secondTensor, "different values", s"values of ${values}")
		}
	}
	
	
	/**
	 * 
	 */
	def runProposedAlgorithm(tensor1: Tensor, tensor2: Tensor, experiment: String, condition: String) = {
		// Coupled decomposition
		val decomposition1 = ALS(tensor1, 3).withMaxIterations(500).withPrintEvery(100).withInitializer(ALS.Initializers.uniform)
		val decomposition2 = ALS(tensor2, 2).withMaxIterations(500).withPrintEvery(100).withInitializer(ALS.Initializers.uniform)
		val coupledDecomposition = new CoupledCP(Array(decomposition1, decomposition2), Array(0, 0)).withThreshold(0.5).withMergingScore(CoupledCP.MergingScores.approximatedWeightedKendallCorrelation(0.5))
		val resultCoupledDecomposition = coupledDecomposition.execute()
		
		// Clusterize
		val clusters = clusterize(resultCoupledDecomposition.A(0)(0))
		println(s"Index for $experiment experiment ${if (condition.trim.nonEmpty) "with " + condition + " " else ""}(proposed algorithm):")
		val eval = evaluateClusters(clusters, resultCoupledDecomposition.A(0)(0))
		ariProposedAlgorithm = ariProposedAlgorithm :+ "%1.3f".format(eval._1)
		dbiProposedAlgorithm = dbiProposedAlgorithm :+ "%1.3f".format(eval._2)

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecomposition.A(0)(0), s"${experiment.capitalize} experiment ${if (condition.trim.nonEmpty) "with " + condition + " " else ""}(proposed algorithm)")
	}
	
	/**
	 * 
	 */
	def runDLAlgorithm(tensor1: Tensor, tensor2: Tensor, experiment: String, condition: String) = {
		// De Lathauwer
		val coupledDecompositionDL = CoupledALS(Array(tensor1, tensor2), 4, Array(CoupledDimension(tensor1, tensor2, Map(0 -> 0)))).withMaxIterations(500).withPrintEvery(100).withInitializer(CoupledALS.Initializers.gaussian)
		val resultCoupledDecompositionDL = coupledDecompositionDL.execute()
		
		// Clusterize
		val clustersDL = clusterize(resultCoupledDecompositionDL.A(0)(0))
		println(s"Index for $experiment experiment ${if (condition.trim.nonEmpty) "with " + condition + " " else ""}(De Lathauwer algorithm):")
		val eval = evaluateClusters(clustersDL, resultCoupledDecompositionDL.A(0)(0))
		ariDLAlgorithm = ariDLAlgorithm :+ "%1.3f".format(eval._1)
		dbiDLAlgorithm = dbiDLAlgorithm :+ "%1.3f".format(eval._2)

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecompositionDL.A(0)(0), s"${experiment.capitalize} experiment ${if (condition.trim.nonEmpty) "with " + condition + " " else ""}(De Lathauwer algorithm)") 
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
		val _matrix = matrix//.map(_ / div)
		val canvas = new Heatmap(yLabel.toArray, xLabel.toArray, _matrix.t.toArray.grouped(matrix.cols).toArray, palette.reverse).canvas()
		canvas.setAxisLabels(s"Factors", s"Elements")
		canvas.setTitle(title)
		canvas.window()
	}
	 
	def clusterize(matrix: DenseMatrix[Double]): List[List[(Int, Double)]] = {
		val vectors = (for (i <- 0 until matrix.cols) yield matrix(::, i)).toList
		Clustering(vectors).run()
	}
	
	def evaluateClusters(clusters: List[List[(Int, Double)]], matrix: DenseMatrix[Double]): (Double, Double) = {
		val randIndex = Clustering.computeRandIndex(clusters, groundTruth)
		println(s"Rand Index:${randIndex}")
		val data = clusters.map(cluster => {
			cluster.map(e => (e._1, matrix(e._1, ::).t))
		})
		val dbIndex = Clustering.computeDaviesBouldinIndex(data)
		println(s"Davies Bouldin:${dbIndex}")
		(randIndex.adjusted, dbIndex.index)
	}
}

/**
 * Clustering method.
 */
class Clustering private(data: List[DenseVector[Double]]) {
	def run(): List[List[(Int, Double)]] = {
		val normalizedVectors = data.map(v => v / max(v))
		(for (i <- 0 until normalizedVectors.head.length) yield {
			var maxValue = 0.0
			var maxCluster = 0
			for (j <- normalizedVectors.indices) {
				val factor = normalizedVectors(j)
				if (factor(i) > maxValue) {
					maxValue = factor(i)
					maxCluster = j
				}
			}
			(maxCluster, (i, data(maxCluster)(i)))
		}).groupBy(_._1).map(e => e._2.map(_._2).toList).toList
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
		// Compute centroids
		val centroids = clusters.map { cluster =>
			cluster.map(_._2).reduce(_ + _) / cluster.size.toDouble
		}

		// Compute average distance of points to their centroid
		val scatters = clusters.zip(centroids).map { case (cluster, centroid) =>
			cluster.map { case (_, point) =>
			norm(point - centroid)
			}.sum / cluster.size.toDouble
		}

		// Compute Davies-Bouldin index
		val index = clusters.indices.map { i =>
			val maxR = clusters.indices.filter(_ != i).map { j =>
				val centroidDistance = norm(centroids(i) - centroids(j))

				if (centroidDistance == 0.0)
					Double.PositiveInfinity
				else
					(scatters(i) + scatters(j)) / centroidDistance
				}.max

			maxR
		}.sum / clusters.size.toDouble

		DaviesBouldinIndex(index, scatters.toArray)
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
