This experiment aims at observing the analytical capabilities of the coupled CANDECOMP/PARAFAC decomposition.

To do so, 2 tensors are built, and 4 experiments are conducted. The first tensor has 3 dimensions of size 30, and contains 3 clusters (the first from elements 1 to 10 on each dimension, the second from elements 11 to 20, and the third from elements 21 to 30). The second tensor has 3 dimensions, the first of size 30 and the second and third of size 10, with 2 clusters (the first from element 0 to 14 on dimension 1 and from elements 1 to 5 on dimensions 2 and 3, the second from element 16 to 30 on dimension 1 and from elements 6 to 10 on dimensions 2 and 3). The tensors are coupled on their first dimension.

The first experiment execute a non-coupled decomposition on the first tensor. The factor matrix corresponding to the fisrt dimension of the tensor is shown below. As expected, only 3 clusters are visible as the second tensor is not used in the decomposition.
![Result of the first experiment](HeatmapSimpleDecomposition.png?raw=true "Result of the first experiment")

The second experiment execute a coupled decomposition on the first and the second tensor. The factor matrix corresponding to the coupled dimension is shown below. This time, 4 clusters can be distinguished: the data of the second tensor allow to "split" the second cluster in 2 clusters.
![Result of the second experiment](HeatmapSimpleCoupledDecomposition.png?raw=true "Result of the second experiment")

The third experiment is similar to the second experiment, except that some elements are removed from the first cluster of the first tensor (from elements 10 to elements 15). We can see in the factor matrix below that the second cluster is now closer to the first tensor, as the only information available are in the second tensor containing a unique cluster from element 1 to 15. 
![Result of the third experiment](HeatmapMissingDataCoupledDecomposition.png?raw=true "Result of the third experiment")

In the final experiment, some noise is added in the first tensor. 50 values are added for each element of the first dimension. The 4 clusters are still cleary appearing, only some minor perturbation in the values of the factor matrix corresponding to each cluster can be observed. 
![Result of the forth experiment](HeatmapNoisyDataCoupledDecomposition.png?raw=true "Result of the forth experiment")

To run the experiments, use the following command:
```scala
scala -classpath lib/*:. CoupledALSExperiments.scala
```

The version of Scala used is 2.12.20.
