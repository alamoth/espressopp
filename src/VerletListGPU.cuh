#include <cuda_runtime.h>
#include "storage/StorageGPU.hpp"
#include <stdio.h>
namespace espressopp {
void verletListBuildDriver(StorageGPU* GS, int n_pt, realG cutsq, int* d_vlPairs, int* d_n_nb, int max_n_nb);

}
