# MuLOT: the Spark tensor CANDECOMP/PARAFAC decomposition
This implementation of the tensor CP decomposition in Spark can run on large scale tensors, containing billions of non-zero elements. Based on CoordinateMatrix and BlockMatrix, it benefits from distributed computing capabilities of Spark, thus allowing efficient processing. It is the ideal tool to perform CP decomposition directly on data, without having to create an intermediate structure used to map a value to its index in the tensor, as it works directly with DataFrame, with values of dimensions of any type. Performance of MuLOT makes it capable to analyze data ranging from small to large-scale.

## Using the Spark decomposition
To use the decomposition, the first step is to create a DataFrame containing a column by dimension, and one more for storing the values of the tensor. Then, the tensor can be created from the DataFrame:
```scala
val tensor = Tensor(df)
```
Note that by default the columns of the dimensions can have any name, and the column of the tensor's values is considered to be called "val". To change that, you can construct a tensor by indicating the name of the column containing the tensor's values:
```scala
val tensor = Tensor(df, "tensorValuesColumnName")
```

When the tensor is built, the CP decomposition can be called directly, with 3 being the wanted rank:
```scala
val cpResult = tensor.runCPALS(3)
```

Some optional parameters are available:
- nbIterations (default 25): the maximum number of iterations ; 
- norm (default CPALS.NORM_L1): the norm to use to normalize the factor matrices (NORM_L1 and NORM_L2 are available), 
- minFms (default 0.99): the convergence limit to stop the iterations (the Factor Match Score is used to determine convergence), 
- checkpoint (default false): set to true to use Spark checkpoints. It can improve performances.

The result of the decomposition is a Map[String, DataFrame], with keys being the original name of each dimension, associated with a DataFrame of 3 columns: the value of the dimension, the rank, and the value found with the CP decomposition.

## Experiments
Our implementation have been tested to compare its execution time with other CP decomposition libraries made for large-scale tensors. The notebooks can be found in the experiments folder. MuLOT outperforms these libraries at large-scale, while being suitable for small and medium tensors analysis.

![Benchmark results](experiments/CPALS_benchmark_dim3.png?raw=true "Benchmark results")
