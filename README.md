# Spark Tensor Decomposition: CP ALS
This implementation of the tensor CP decomposition in Spark can run on large scale tensors, containing billions of non-zero elements. Based on CoordinateMatrix and BlockMatrix, it benefits from distributed computing capabilities of Spark, thus allowing efficient processing.

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

The result of the decomposition is a Map[String, DataFrame], with keys being the original name of each dimension, associated with a DataFrame of 3 columns: the value of the dimension, the rank, and the value found with the CP decomposition.

## Experiments
Our implementation have been tested to estimate its execution time with tensors having different number of dimensions, dimension size and sparsity. The notebooks can be found in the experiments folder.
