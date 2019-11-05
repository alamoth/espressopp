#include <stdio.h>
#include <cuda.h>
#include "storage/StorageGPU.cuh"

__global__ void 
verletListBuild(  const int nPart,
                  const int nCells,
                  const realG3* __restrict__ pos,
                  const int* __restrict__ type,
                  const bool* __restrict__ real,
                  const int* __restrict__ cellParticles, 
                  const int* __restrict__ cellOffsets,
                  const int* __restrict__ cellNeighbors,
                  const realG cutoff,
                  int** vl1,
                  int** vl2,
                  int* numN
                ){

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!real[idx]) return;
  if (idx >= nPart) return;

  realG3 p_dist;
  realG3 p_pos1 = pos[idx];
  int neighborC = 0;
  for(int i = 0; i < 27; ++i){
    currentCellId = cellNeighbors[p_cellId * 27 + i];
    for(int j = 0; j < cellParticles[currentCellId]; ++j){
      if(cellOffsets[currentCellId] + j != idx){
        realG3 p_pos2 = pos[cellOffsets[currentCellId] + j];
        p_dist.x = p_pos1.x - p_pos2.x;
        p_dist.y = p_pos1.y - p_pos2.y;
        p_dist.z = p_pos1.z - p_pos2.z;
        real distSqr =  p_dist.x * p_dist.x + p_dist.y * p_dist.y + p_dist.z * p_dist.z;
        if(distSqr <= (cutoff * cutoff){

        }
      }
    }
  }

}
