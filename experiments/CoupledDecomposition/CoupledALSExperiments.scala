import breeze.linalg._

import mulot.local.Tensor
import mulot.local.tensordecomposition._
import mulot.local.tensordecomposition.cp.ALS
import mulot.local.tensordecomposition.cp.ALS._
import mulot.local.tensordecomposition.cp.CoupledALS
import mulot.local.tensordecomposition.cp.CoupledALS._
import mulot.core.tensordecomposition.CoupledDimension

import java.awt.Color
import collection.JavaConverters._

import smile.data._
import smile.data.`type`._
import smile.plot.swing._

object CoupledALSExperiments {
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
	def createNoise(nb: Int, dimension1: Range, dimension2: Array[Int], dimension3: Array[Int]): Map[Array[Int], Double] = {
		val rand = new scala.util.Random
		var data = Map.empty[Array[Int], Double]
		for (dim1 <- dimension1) {
			for (i <- 0 until nb) {
				data += Array(dim1, dimension2(rand.nextInt(dimension2.length)), dimension3(rand.nextInt(dimension3.length))) -> rand.nextInt(10).toDouble
			}
		}
		data
	}

	// Execute with "scala -classpath lib/*:. CoupledALSExperiments.scala"
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
		
		baselineDecomposition(mainTensor, 3)
		simpleExperiment(mainTensor, secondTensor, 3)
		missingDataExperiment(mainTensorData, secondTensor, 3)
		noisyDataExperiment(mainTensorData, secondTensor, 3, .5)
	}
	
	/**
	 * Execute a CP decomposition on the main tensor.
	 */
	def baselineDecomposition(mainTensor: Tensor, rank: Int): Unit = {
		// Baseline decomposition
		val baselineDecomposition = ALS(mainTensor, rank).withInitializer(ALS.Initializers.hosvd)
		val resultBaselineDecomposition = baselineDecomposition.execute()

		// Produce vizualisation of result
		plotMatrix(resultBaselineDecomposition.A(0))
	}
	
	/**
	 * Execute a coupled CP decomposition on the main tensor and the second tensor. 
	 */
	def simpleExperiment(mainTensor: Tensor, secondTensor: Tensor, rank: Int): Unit = {
		// Coupled decomposition
		val coupledDecomposition = CoupledALS(Array(mainTensor, secondTensor), rank, Array(CoupledDimension(mainTensor, secondTensor, Map(0 -> 0)))).withInitializer(CoupledALS.Initializers.hosvd)
		val resultCoupledDecomposition = coupledDecomposition.execute()

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecomposition.A(0)(0))
	}
	
	/**
	 * Remove half the elements of a cluster only in one tensor to see how the coupled decomposition deals with missing data.
	 */
	def missingDataExperiment(mainTensorData: Map[Array[Int], Double], secondTensor: Tensor, rank: Int): Unit = {
		val removedKeys = createCluster(10 until 15, 0 until 30, 0 until 30).keys.toArray
		val newMainTensorData = mainTensorData.filterKeys(k => !removedKeys.exists(k2 => k2.sameElements(k)))
		val newMainTensor = Tensor.fromIndexedMap(newMainTensorData, 3, Array(30, 30, 30), Array("dimension1", "dimension2", "dimension3"))
		
		// Coupled decomposition
		val coupledDecomposition = CoupledALS(Array(newMainTensor, secondTensor), rank, Array(CoupledDimension(newMainTensor, secondTensor, Map(0 -> 0)))).withInitializer(CoupledALS.Initializers.hosvd)
		val resultCoupledDecomposition = coupledDecomposition.execute()

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecomposition.A(0)(0))
	}
	 
	/**
	 * Add noise in the main tensor to see how the decomposition performs. 
	 */
	def noisyDataExperiment(mainTensorData: Map[Array[Int], Double], secondTensor: Tensor, rank: Int, noisePercent: Double): Unit = {
		var addedEntries = createNoise((100 * noisePercent).toInt, 0 until 10, (10 until 30).toArray, (10 until 30).toArray)
		addedEntries ++= createNoise((100 * noisePercent).toInt, 10 until 20, ((0 until 10) ++ (20 until 30)).toArray, ((0 until 10) ++ (20 until 30)).toArray)
		addedEntries ++= createNoise((100 * noisePercent).toInt, 20 until 30, (0 until 20).toArray, (0 until 20).toArray)
		val newMainTensorData = mainTensorData ++ addedEntries
		val newMainTensor = Tensor.fromIndexedMap(newMainTensorData, 3, Array(30, 30, 30), Array("dimension1", "dimension2", "dimension3"))
		
		// Coupled decomposition
		val coupledDecomposition = CoupledALS(Array(newMainTensor, secondTensor), rank, Array(CoupledDimension(newMainTensor, secondTensor, Map(0 -> 0)))).withInitializer(CoupledALS.Initializers.hosvd)
		val resultCoupledDecomposition = coupledDecomposition.execute()

		// Produce vizualisation of result
		plotMatrix(resultCoupledDecomposition.A(0)(0))
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
	def plotMatrix(matrix: DenseMatrix[Double]): Unit = {
		heatmapPlot(matrix)
	 }
	 
	 def tSnePlot(matrix: DenseMatrix[Double]): Unit = {
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
			canvas.setAxisLabels(s"Factor $f1", s"Factor $f2")
			grid.add(canvas.panel)
		}
		grid.window()
	 }
	 
	 def heatmapPlot(matrix: DenseMatrix[Double]): Unit = {
	 	val n = 32
		val palette = new Array[Color](n)
        for (i <- 0 until (n / 2)) {
            palette(i) = new Color((n - (i * 2)).toFloat / n.toFloat, 0.0f, 0.0f, 0.8f)
            palette(n - i - 1) = new Color(0.0f, (n - (1.0f + i * 2)).toFloat / n.toFloat, (n - (1.0f + i * 2)).toFloat / n.toFloat, 0.8f)
        }
        val xLabel = for (i <- 0 until matrix.cols) yield s"${i + 1}"
        val yLabel = for (i <- 0 until 30) yield s"${i + 1}"
        val matrixMax = max(matrix)
        val matrixMin = min(matrix)
        val _matrix = matrix.map(v => if (v < 0.0) v / -matrixMin else v / matrixMax)
		val canvas = new Heatmap(yLabel.toArray, xLabel.toArray, _matrix.t.toArray.grouped(matrix.cols).toArray, palette.reverse).canvas()
		canvas.setAxisLabels(s"Factors", s"Elements")
		canvas.window()
	 }
}
