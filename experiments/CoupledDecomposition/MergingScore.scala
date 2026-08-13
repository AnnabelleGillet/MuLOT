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

object MergingScore {
	val groundTruth = (for (i <- 0 until 30) yield {
		if (i < 10) 0
		else if (i < 15) 1
		else if (i < 20) 2
		else 3
	}).toArray
	
	// For storing result
	var result = scala.collection.mutable.Map[String, Map[String, List[Double]]]()
	var condition = ""
	var currentVector = 0
	
	def computeAllMergingScore(mergedVector: DenseVector[Double], factors: List[DenseVector[Double]]): Double = {

		val kendall = CoupledCP.MergingScores.kendallCorrelation(mergedVector, factors)
		val weightedKendall = CoupledCP.MergingScores.weightedKendallCorrelation(mergedVector, factors)
		val approximatedWeightedKendall = CoupledCP.MergingScores.approximatedWeightedKendallCorrelation(0.5)(mergedVector, factors)
		val spearman = CoupledCP.MergingScores.spearmanCorrelation(mergedVector, factors)
		val cosine = CoupledCP.MergingScores.cosineSimilarity(mergedVector, factors)
		result(condition) = result(condition) + ("Kendall" -> (result(condition).getOrElse("Kendall", List[Double]()) :+ kendall))
		result(condition) = result(condition) + ("Weighted Kendall" -> (result(condition).getOrElse("Weighted Kendall", List[Double]()) :+ weightedKendall))
		result(condition) = result(condition) + ("Approximated Weighted Kendall" -> (result(condition).getOrElse("Approximated Weighted Kendall", List[Double]()) :+ approximatedWeightedKendall))
		result(condition) = result(condition) + ("Spearman" -> (result(condition).getOrElse("Spearman", List[Double]()) :+ spearman))
		result(condition) = result(condition) + ("Cosine similarity" -> (result(condition).getOrElse("Cosine similarity", List[Double]()) :+ cosine))
		currentVector += 1
		if ((weightedKendall > 0.5 && approximatedWeightedKendall < 0.5) || (weightedKendall < 0.5 && approximatedWeightedKendall > 0.5)) {
			println(s"PB! $weightedKendall != $approximatedWeightedKendall")
		}
		weightedKendall
	}
		
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

	// Execute with "scala -classpath lib/*:. MergingScore.scala"
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

		// Noise
		for (n <- List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)) {
			condition = s"Noise ${n.toInt}%"
			currentVector = 0
			result = result + (condition -> Map[String, List[Double]]())
			noisyDataExperiment(mainTensorData, secondTensorData, 4, n / 100)
		}
		plotResult(result, "Noise", "Noise.png")
		
		// Missing data
		result = scala.collection.mutable.Map[String, Map[String, List[Double]]]()
		for (n <- List(10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0)) {
			condition = s"Missing ${n.toInt}%"
			currentVector = 0
			result = result + (condition -> Map[String, List[Double]]())
			missingDataExperiment(mainTensorData, secondTensorData, 4, n / 100)
		}
		plotResult(result, "Missing data", "MissingData.png")
		
		// Different cluster sizes
		result = scala.collection.mutable.Map[String, Map[String, List[Double]]]()
		currentVector = 0
		for (n <- 1 until 10) {
			condition = s"${10 - n} elements in common"
			val noise = 5
			result = result + (condition -> Map[String, List[Double]]())
			secondTensorData = Map[Array[Int], Double]()
			secondTensorData ++= createCluster(n until 15, 0 until 5, 0 until 5)
			secondTensorData ++= createCluster(15 until 30, 5 until 10, 5 until 10)
			simpleExperiment(mainTensor, Tensor.fromIndexedMap(secondTensorData, 3, Array(30, 10, 10), Array("dimension1", "dimension2", "dimension3")), 4)
			currentVector += 1
		}
		plotResult(result, "Varying cluster size", "DifferentClusterSizes.png")
		
		for ((key, value) <- result) {
			print(s"$key: ")
			println(value.mkString(", "))
		}
	}
	
	/**
	 * Execute a coupled CP decomposition on the main tensor and the second tensor. 
	 */
	def simpleExperiment(mainTensor: Tensor, secondTensor: Tensor, rank: Int): Unit = {
		// Coupled decomposition
		val decomposition1 = ALS(mainTensor, 3).withMaxIterations(500).withPrintEvery(100)
		val decomposition2 = ALS(secondTensor, 2).withMaxIterations(500).withPrintEvery(100)
		val coupledDecomposition = new CoupledCP(Array(decomposition1, decomposition2), Array(0, 0)).withMergingScore(computeAllMergingScore)
		
		val resultCoupledDecomposition = coupledDecomposition.execute()
	}
	
	/**
	 * Remove elements of clusters only in one tensor to see how the coupled decomposition deals with missing data.
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
		
		// Coupled decomposition
		val decomposition1 = ALS(newMainTensor, 3).withMaxIterations(500).withPrintEvery(100)
		val decomposition2 = ALS(newSecondTensor, 2).withMaxIterations(500).withPrintEvery(100)
		val coupledDecomposition = new CoupledCP(Array(decomposition1, decomposition2), Array(0, 0)).withMergingScore(computeAllMergingScore)
		val resultCoupledDecomposition = coupledDecomposition.execute()
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
		val decomposition1 = ALS(newMainTensor, 3).withMaxIterations(500).withPrintEvery(100)
		val decomposition2 = ALS(newSecondTensor, 2).withMaxIterations(500).withPrintEvery(100)
		val coupledDecomposition = new CoupledCP(Array(decomposition1, decomposition2), Array(0, 0)).withMergingScore(computeAllMergingScore)
		val resultCoupledDecomposition = coupledDecomposition.execute()
	}
	
	
	// Data structure for vizualisation
	val struct = DataTypes.struct(new StructField("factor1", DataTypes.DoubleType), new StructField("factor2", DataTypes.DoubleType), new StructField("cluster", DataTypes.StringType))
	case class Entry(factor1: Double, factor2: Double, cluster: String) extends smile.data.Tuple {
		override def schema(): StructType = struct
		override def get(x: Int): Object = if (x == 0) factor1.asInstanceOf[Object] else if (x == 1) factor2.asInstanceOf[Object] else if (x == 2) cluster.asInstanceOf[Object] else null
	}
	 
	/*
	 * Produce a visualisation for the metrics.
	 */
	def plotResult(res: scala.collection.mutable.Map[String, Map[String, List[Double]]], title: String, saveFile: String): Unit = {
		val grid = new javax.swing.JPanel(new java.awt.GridLayout(5, 2))
		val limit = LinePlot.of(Array[Array[Double]](Array(1.5, -1.0), Array(1.5, 1.5)), Line.Style.SOLID, Color.BLACK)
		val marks = Array('O', 'S', 'Q', '*', 'x', 'o', '@', '#', 's', 'q', '.', '+', '-', '|')
		val colors = Array[Color](Color.ORANGE, Color.GREEN, Color.RED, Color.BLUE, Color.MAGENTA)
		val legends = new Array[(Color, String)](5)
		var sortedConditions = res.keys.toList.sorted
		if (sortedConditions(0).contains("elements")) sortedConditions = sortedConditions.reverse
		for (condition <- sortedConditions) {
			val metrics = List("Cosine similarity", "Spearman", "Kendall", "Weighted Kendall", "Approximated Weighted Kendall")
			val sortedIndex = res(condition)("Weighted Kendall").zipWithIndex.sortWith((v1, v2) => v1._1 < v2._1).map(_._2)
			val s = metrics.size
			val points = new Array[Point](s)
			var i = 0
			val canvas = limit.canvas()
			for (metric <- metrics) {
				val values = res(condition)(metric)
				val data = sortedIndex.map(values(_)).toArray
				canvas.add(Line.of(Line.zipWithIndex(data), Line.Style.SOLID, colors(i)))
				points(i) = new Point(Line.zipWithIndex(data), marks(i), colors(i))
				canvas.add(points(i))
				legends(i) = (colors(i), metric)
				i += 1
			}
			
			canvas.extendLowerBound(Array(0, -0.1))
			canvas.extendUpperBound(Array(5, 1.1))
			canvas.setTitle(condition)
			canvas.setAxisLabels(s"Factor", s"Value")
			grid.add(canvas.panel)
		}
		val legendPanel = new LegendPanel(legends)
		legendPanel.setBackground(Color.WHITE)
		grid.add(legendPanel, java.awt.BorderLayout.EAST)
		
		val frame = new javax.swing.JFrame(title)
		frame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE)
		frame.setContentPane(grid)
		frame.setSize(1200, 1600)
		frame.setLocationRelativeTo(null)
		frame.setVisible(true)
		
		import java.awt.image.BufferedImage
		grid.setSize(1200, 1600)
		grid.doLayout
		val image = new BufferedImage(1200, 1600, BufferedImage.TYPE_INT_ARGB)
		val g = image.createGraphics()
		grid.validate()
		grid.doLayout()
		grid.printAll(g)
		g.dispose()
		javax.imageio.ImageIO.write(image, "png", new java.io.File(saveFile))
	}
}

class LegendPanel(legends: Array[(Color, String)]) extends javax.swing.JPanel {
    setLayout(new javax.swing.BoxLayout(this, javax.swing.BoxLayout.Y_AXIS))
    setBackground(Color.WHITE)

    for (legend <- legends) {
        val row = new javax.swing.JPanel(new java.awt.FlowLayout(java.awt.FlowLayout.CENTER))
        row.setBackground(Color.WHITE)

        val colorBox = new javax.swing.JPanel()
        colorBox.setPreferredSize(new java.awt.Dimension(12, 12))
        colorBox.setBackground(legend._1)

        val text = new javax.swing.JLabel(legend._2)
        text.setBackground(Color.WHITE)

        row.add(colorBox)
        row.add(text)

        add(row)
    }
}
