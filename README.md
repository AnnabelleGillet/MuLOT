# MuLOT: the Scala tensor decomposition library
MuLOT provides tensor decompositions for Scala. It has a distributed and a local implementation, that allow to compute efficiently the decomposition, on small and large tensors with billions of non-zero elements. The distributed implementation is based on Spark. 

It is also data centric, and avoid the direct manipulation of mathematical structures by the user. It can provide the result with meaningful values rather than with only integer indexes. 

The optimization of the library is a major concern, and it outperforms the state of the art of large-scale tensor decomposition libraries.

## Importing the library
Put the chosen jar (local or distributed) in the `lib` directory, at the root of your Scala project. If the distributed version is chosen, Spark must be imported in `built.sbt`:

```scala
libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.2.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.2.0"
```

## Building tensors
In MuLOT, tensors are sparse structures. The have _n_ dimensions, and the values of a tensor are indexed by a combination of one value for each dimension.

### Local 
First, the following import statement is needed:
 ```scala
 import mulot.local.Tensor
 ```
The local implementation build a tensor from:
 - a Map, with the key as an Array of any type, and the value as a Double;
 - an order (the number of dimensions of the tensor);
 - an Array of String, containing the name of the dimensions, given in the same order as the keys.
```scala
val tensor = Tensor(data, order, dimensionsName)
```

### Distributed
First, the following import statement is needed:
```scala
import mulot.distributed.Tensor
```
An implicit SparkSession is required when working with the distributed implementation. 
To build a distributed tensor, the first step is to create a DataFrame containing a column by dimension, and one more for storing the values of the tensor. Then, the tensor can be created from the DataFrame:
```scala
val tensor = Tensor(df)
```
Note that by default the columns of the dimensions can have any name, and the column of the tensor's values is considered to be called "val". To change that, you can construct a tensor by indicating the name of the column containing the tensor's values:
```scala
val tensor = Tensor(df, "tensorValuesColumnName")
```

## The CANDECOMP/PARAFAC decomposition
Import the decomposition:
```scala
import mulot.local.tensordecomposition.cp // For the local implementation
import mulot.distributed.tensordecomposition.cp // For the distributed implementation
```

The use of the decompositions is the same in the local and the distributed implementations. The tensor on which it is applied and the rank (as an Int) are required:
```scala
val cp = CP(tensor, rank)
```
Some optional parameters are available:
```scala
val cp = CP(tensor, rank)
    .withMaxIterations(50) // the maximum number of iterations (default 25)
    .withMinFms(0.95) // The Factor Match Score threshold used to stop the iterations (default 0.99)
    .withNorm(Norms.L1) // The norm to use on the factor matrices (default L2)
    .withInitializer(Initializers.hosvd) // The method used to initialize the factor matrices (default gaussian)
    .withComputeCorcondia(true) // To decide if CORCONDIA must be computed on the result (default false)
```
CORCONDIA is the [core consistency diagnostic](https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/pdf/10.1002/cem.801), and can be used to evaluate the quality of the decomposition. 

Once the decomposition is set, it is run with:
```scala
val kruskal = cp.execute()
```
To obtain meaningful values in the result rather than integer indexes, it can be converted:
```scala
val result = kruskal.toExplicitValues()
```
The return type of this method is different in the local and in the distributed implementation:
- **In the local implementation**, the result is a Map[String, Array[Map[Any, Double]]], with the String keys corresponding to the name of the dimensions, the index of the Array correspond to the rank for this dimension, and the Map[Any, Double] associates each element of the dimension to its value obtained with the CP decomposition;
- **In the distributed implementation**, the result is a Map[String, DataFrame], with keys being the original name of each dimension, associated with a DataFrame of 3 columns: the value of the dimension, the rank, and the value found with the CP decomposition.

## Experiments
Our implementation of the CP decomposition has been tested to compare its execution time with other CP decomposition libraries made for large-scale tensors. The notebooks can be found in the `experiments` folder. MuLOT outperforms these libraries at large-scale, while being suitable for small and medium tensors analysis. TensorLy is used as a reference of non-distributed library. 3-order tensors were used for this experiment. 

![Benchmark results](experiments/CPALS_benchmark_dim3.png?raw=true "Benchmark results")

## Roadmap
- The Tucker decomposition is under development, with two algorithms : HOOI to enforce the orthogonality constraint and HALS-NTD to enforce the non-negativity constraint;
- An extended documentation on the tensor decompositions and the library will be released.

## To cite the work
GILLET, Annabelle, LECLERCQ, Éric, et CULLOT, Nadine. MuLOT: Multi-level Optimization of the Canonical Polyadic Tensor Decomposition at Large-Scale. In : European Conference on Advances in Databases and Information Systems. Springer, Cham, 2021. p. 198-212.
