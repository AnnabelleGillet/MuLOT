package mulot.local.tensordecomposition.cp

import breeze.linalg.{DenseMatrix, DenseVector, cosineDistance, max, min, sum}
import mulot.core.tensordecomposition.cp.Norms
import mulot.local.Tensor
import scribe.Logging

import scala.collection.parallel.CollectionConverters.ImmutableSeqIsParallelizable
import scala.util.control.Breaks.break

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
	
	object MergingScores {
		def entropy(mergedVector: DenseVector[Double], factors: List[DenseVector[Double]]): Double = {
			- mergedVector.mapValues(v => v * math.log(v)).data.sum / mergedVector.length
		}
		
		def cosineSimilarity(mergedVector: DenseVector[Double], factors: List[DenseVector[Double]]): Double = {
			(for (factor <- factors) yield cosineDistance(mergedVector, factor)).min
		}
		
		def kendallCorrelation(mergedVector: DenseVector[Double], factors: List[DenseVector[Double]]): Double = {
			(for (factor <- factors) yield {
				(2.0 / (mergedVector.length * (mergedVector.length - 1))) * (for (i <- mergedVector.data.indices; j <- i + 1 until mergedVector.length) yield {
					mergedVector(i).compare(mergedVector(j)) * factor(i).compare(factor(j))
				}).sum
			}).min
		}
		
		def weightedKendallCorrelation(_mergedVector: DenseVector[Double], factors: List[DenseVector[Double]]): Double = {
			// Sort merged vector to not have to compute the order comparison for its elements
			val sortedMergedVector = _mergedVector.iterator.toArray.sortBy(_._2).reverse
			val mergedVector = sortedMergedVector.map(_._2)
			val logMergedVector = mergedVector.map(math.log) // Precompute log
			
			var score = 1.0
			for (_factor <- factors.par) {
				// Follow the order of elements of the merged vector
				val factor = sortedMergedVector.map(v => _factor(v._1))
				val logFactor = factor.map(math.log) // Precompute log
				var res = 0.0
				var fact = 0.0
				var index = mergedVector.indices.tail
				
				for (i <- mergedVector.indices) {
					val logMergedVectorI = logMergedVector(i)
					
					for (j <- index) {
						val logFactorJ = logFactor(j)
						// Compute the weight
						var weight = (mergedVector.length - i) * mergedVector(i) * math.max(1.0 / mergedVector.length, logMergedVectorI - logFactorJ)
						
						if (weight.isInfinity) weight = (mergedVector.length - i) * mergedVector(i)
						else if (weight.isNaN) weight = 1.0
						
						res += weight * weight * (if (factor(i) >= factor(j)) 1.0 else -1.0)
						fact += weight * weight
					}
					if (index.nonEmpty) {
						index = index.tail
					}
				}
				score.synchronized {
					if (score > (res / fact)) {
						score = res / fact
					}
				}
			}
			mergedVector(0) * score
		}
		
		/**
		 * Fast approximation of the Weighted Kendall Correlation. Stop the computation if the result will not change
		 * regarding the threshold.
		 *
		 * @param threshold
		 * @param _mergedVector
		 * @param factors
		 * @return
		 */
		def approximatedWeightedKendallCorrelation(threshold: Double, print: Boolean = false)(_mergedVector: DenseVector[Double], factors: List[DenseVector[Double]]): Double = {
			// Sort merged vector to not have to compute the order comparison for its elements
			val sortedMergedVector = _mergedVector.iterator.toArray.sortBy(_._2).reverse
			val mergedVector = sortedMergedVector.map(_._2)
			
			if (mergedVector(0) > threshold) {
				val realThreshold = threshold / mergedVector(0)
				val logMergedVector = mergedVector.map(math.log) // Precompute log
				
				var score = 1.0
				
				for (_factor <- factors.par) {
					// Follow the order of elements of the merged vector
					val factor = sortedMergedVector.map(v => _factor(v._1))
					val logFactor = factor.map(math.log) // Precompute log
					
					val logFactorMin = new Array[Double](logFactor.length)
					logFactorMin(logFactor.length - 1) = logFactor.last
					
					var k = logFactor.length - 2
					while (k >= 0) {
						logFactorMin(k) = math.min(logFactor(k), logFactorMin(k + 1))
						k -= 1
					}
					
					var index = mergedVector.indices.tail
					
					var res = 0.0
					var fact = 0.0
					
					scala.util.control.Breaks.breakable {
						for (i <- mergedVector.indices) {
							val logMergedVectorI = logMergedVector(i)
							if (mergedVector(i) < threshold && i < mergedVector.length - 1 && logMergedVectorI - logFactorMin(i + 1) >= 0) {
								var maxApproximatedScore = (mergedVector.length - i) * mergedVector(i) * math.max(1.0 / mergedVector.length, logMergedVectorI - logFactorMin(i + 1))
								maxApproximatedScore *= maxApproximatedScore
								maxApproximatedScore *= (mergedVector.length - i) * index.length
								if ((((res - maxApproximatedScore) / (fact + maxApproximatedScore)) > realThreshold) || // Already above threshold and can't get down
									(((res + maxApproximatedScore) / (fact + maxApproximatedScore)) < realThreshold)) { // Already below threshold and can't get up
									if (print) {
										logger.info(s"Break at $i for $maxApproximatedScore added to $res / $fact between (${(res - maxApproximatedScore) / (fact + maxApproximatedScore)}) and (${(res + maxApproximatedScore) / (fact + maxApproximatedScore)}) to $realThreshold")
									}
									break
								}
							}
							
							for (j <- index) {
								val logFactorJ = logFactor(j)
								// Compute the weight
								var weight = (mergedVector.length - i) * mergedVector(i) * math.max(1.0 / mergedVector.length, logMergedVectorI - logFactorJ)
								
								if (weight.isInfinity) weight = (mergedVector.length - i) * mergedVector(i)
								else if (weight.isNaN) weight = 1.0
								
								res += weight * weight * (if (factor(i) >= factor(j)) 1.0 else -1.0)
								fact += weight * weight
							}
							if (index.nonEmpty) {
								index = index.tail
							}
						}
						
						score.synchronized {
							if (score > (res / fact)) {
								score = res / fact
							}
						}
					}
				}
				mergedVector(0) * score
			} else {
				mergedVector(0)
			}
		}
		
		def spearmanCorrelation(mergedVector: DenseVector[Double], factors: List[DenseVector[Double]]): Double = {
			def toRanks(vector: DenseVector[Double]): DenseVector[Double] = {
				DenseVector(vector.iterator.toList.sortWith((v1, v2) => v1._2 > v2._2).zipWithIndex.sortWith(_._1._1 < _._1._1).map(_._2.toDouble).toArray)
			}
			
			val mergedVectorRanks = toRanks(mergedVector)
			
			(for (factor <- factors) yield {
				val factorRanks = toRanks(factor)
				1 - ((6 * sum((mergedVectorRanks - factorRanks).mapValues(v => v * v))) / (factor.length * (factor.length * factor.length - 1)))
			}).min
		}
	}
}

class CoupledCP(val decompositions: Array[ALS], commonDimensions: Array[Int]) extends mulot.core.tensordecomposition.cp.ALS[Tensor, Array[DenseMatrix[Double]], Array[Map[String, Array[Map[Any, Double]]]]]
	with Logging {
	
	override type Return = CoupledCP
	override type LambdaType = Array[Double]
	override protected var rank: Int = 0
	protected var threshold: Double = 0.5
	override private[mulot] var tensor: Tensor = null
	override private[mulot] var convergenceMethod: (Kruskal, Kruskal, Boolean) => Double = null
	private[mulot] var mergingScore: (DenseVector[Double], List[DenseVector[Double]]) => Double = CoupledCP.MergingScores.approximatedWeightedKendallCorrelation(this.threshold)
	
	var mergingScores: List[Double] = List[Double]()
	
	def withThreshold(threshold: Double): Return = {
		val newObject = this.copy()
		newObject.threshold = threshold
		newObject
	}
	
	def withMergingScore(mergingScore: (DenseVector[Double], List[DenseVector[Double]]) => Double): Return = {
		val newObject = this.copy()
		newObject.mergingScore = mergingScore
		newObject
	}
	
	override protected def internalCopy(): Return = {
		val newDecomposition = new CoupledCP(decompositions, commonDimensions)
		newDecomposition.threshold = this.threshold
		newDecomposition.mergingScore = this.mergingScore
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
		val begin = System.currentTimeMillis()
		val newFactorVectors = for (decomposition <- decompositions) yield (for (_ <- 0 until decomposition.tensor.order) yield List[DenseVector[Double]]()).toArray
		
		// Merge factors
		val ranksAdvance = Array.fill[Int](decompositions.length){0}
		var vectorsList = List[(Array[Int], DenseVector[Double])]()
		var over = false
		while (!over) {
			// Get factors of individual decompositions
			val factorsToMerge = (for (i <- ranksAdvance.indices) yield {
				val vector = decompositionResults(i).A(commonDimensions(i))(::, ranksAdvance(i))
				// Normalize vector
				vector / max(vector)
			}).toList
			
			// Merge vector
			val mergedVector = factorsToMerge.reduce(min(_, _)).mapValues(x => if (x.isNaN) 0.0 else x)
			
			// Compute score
			val score = mergingScore(mergedVector, factorsToMerge)
			mergingScores :+= score
			
			if (score > threshold) {
				vectorsList :+= (ranksAdvance.clone(), mergedVector)
			}
			
			// Update ranks advance
			var i = 0
			var ok = false
			while (!ok && i < decompositions.length) {
				ranksAdvance(i) += 1
				if (ranksAdvance(i) >= decompositions(i).rank) {
					ranksAdvance(i) = 0
					i += 1
				} else {
					ok = true
				}
			}
			if (!ok) {
				over = true
			}
		}
		
		logger.info(s"Similarity among ${mergingScores.mkString(", ")}")
		
		// Add remaining vectors to result
		for ((dimensionsIndices, vector) <- vectorsList) {
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
			for (vectors <- result if vectors.nonEmpty) yield {
				val matrix = DenseMatrix.zeros[Double](vectors.head.length, vectors.length)
				for (i <- vectors.indices) {
					matrix(::, i) := vectors(i)
				}
				matrix
			}
		}
		logger.info(s"Merging of vectors computed in ${(System.currentTimeMillis() - begin).toDouble / 1000.0}s")
		
		val newRank = newFactorMatrices(0)(0).cols
		val finalDecompositionResults = for (i <- decompositions.indices) yield {
			val decomposition = decompositions(i)
			
			// Compute lambda
			val lambdas = Array.fill(newRank){0.0}
			if (decomposition.norm == Norms.L2) {
				newFactorMatrices(i)(commonDimensions(i)).foreachPair { case ((_, j), v) => lambdas(j) += v * v }
				for (j <- 0 until newRank) {
					lambdas(j) = math.sqrt(lambdas(j))
				}
			} else {
				newFactorMatrices(i)(commonDimensions(i)).foreachPair { case ((_, j), v) => lambdas(j) += math.abs(v) }
			}
			// Normalize factor matrices
			for (j <- 0 until newRank) {
				newFactorMatrices(i)(commonDimensions(i))(::, j) := newFactorMatrices(i)(commonDimensions(i))(::, j) *:* (1 / lambdas(j))
			}
			
			val newDecomposition = decomposition
				.withInitializer(ALS.Initializers.fixed(newFactorMatrices(i)))
				.withFixedDimensions(decomposition.fixedDimensions :+ commonDimensions(i))
				.withRank(newRank)
			newDecomposition.execute()
		}
		val finalFactorMatrices = (for (finalDecompositionResult <- finalDecompositionResults) yield finalDecompositionResult.A).toArray
		val finalLambdas: Array[Array[Double]] = finalDecompositionResults.map(_.lambdas).toArray
		
		Kruskal(finalFactorMatrices, finalLambdas, None)
	}
}
