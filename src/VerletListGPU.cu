#include "VerletListGPU.cuh"
#include <assert.h>
//#ifdef __NVCC__

namespace espressopp {

__global__ void 
VerletListBuild(  const int nPart,
                  const int nCells,
                  const realG3* __restrict__ pos,
                  const int* __restrict__ cellId,
                  const int* __restrict__ type,
                  const bool* __restrict__ real,
                  const int* __restrict__ cellParticles, 
                  const int* __restrict__ cellOffsets,
                  const int* __restrict__ cellNeighbors,
                  const realG cutsq,
                  int* vl,
                  int* num_nb,
                  int max_n_nb
                ){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= nPart) return;
  if (!real[idx]) return;

  realG3 p_dist;
  realG3 p_pos1 = pos[idx];
  int p_num_nb = 0;
  int p_cellId = cellId[idx];

  for(int i = 0; i < 27; ++i){
    int currentCellId = cellNeighbors[p_cellId * 27 + i];
    for(int j = 0; j < cellParticles[currentCellId]; ++j){
      int memIdx = cellOffsets[currentCellId] + j;
      if(memIdx != idx){
        realG3 p_pos2 = pos[memIdx];
        p_dist.x = p_pos1.x - p_pos2.x;
        p_dist.y = p_pos1.y - p_pos2.y;
        p_dist.z = p_pos1.z - p_pos2.z;
        realG distSqr =  p_dist.x * p_dist.x + p_dist.y * p_dist.y + p_dist.z * p_dist.z;
        if(distSqr <= cutsq){
          assert(p_num_nb < max_n_nb);
          assert(memIdx >= 0);
          vl[p_num_nb * nPart + idx] = memIdx;
          p_num_nb++;
        }
      }
    }
  }
  num_nb[idx] = p_num_nb;
}


void verletListBuildDriver(StorageGPU* GS, int n_pt, realG cutsq, int* d_vlPairs, int* d_n_nb, int max_n_nb){

    int threadsPerBlock = 256;
    VerletListBuild<<<(n_pt + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        n_pt,
        GS->numberLocalCells,
        GS->d_pos,
        GS->d_cellId,
        GS->d_type,
        GS->d_real,
        GS->d_particlesCell,
        GS->d_cellOffsets,
        GS->d_cellNeighbors,
        cutsq,
        d_vlPairs,
        d_n_nb,
        max_n_nb);
    cudaDeviceSynchronize(); CUERR
  }
}
// #endif