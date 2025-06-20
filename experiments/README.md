# Benchmark experiments
Each notebook performs a CANDECOMP/PARAFAC decomposition on artificially created tensors, in order to compare the execution time of MuLOT against other libraries of the state of the art. The tested libraries are:
- [BigTensor](https://datalab.snu.ac.kr/bigtensor/index.php)
- [CSTF](https://github.com/ZacBlanco/cstf)
- [HaTen2](https://datalab.snu.ac.kr/haten2/)
- [SamBaTen](https://github.com/lucasjliu/SamBaTen-Spark)
- [TensorLy](https://tensorly.org/stable/index.html)

# Stratification method
The stratification method gathers clusters of similar intensity in each stratum. To do so, the CANDECOMP/PARAFAC decomposition is executed with a rank R, the R clustes found (composed of the elements of each dimension that contribute the most to each rank) are removed from the input tensor, and the decomposition is executed anew to build the next stratum. Each new stratum is formed with clusters that have a lighter signal intensity than the previous stratum, so this method can be used to find weak signals in a dataset. 

# Coupled decompositions
Coupled decompositions allow to perform a decomposition simultaneously on several tensors that share at least one common dimension. The conducted experiments show the analytical capabilities of the coupled CANDECOMP/PARAFAC decomposition compared to the standard CANDECOMP/PARAFAC decomposition, and when dealing with missing or noisy data. 
