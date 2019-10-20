/*
  Copyright (C) 2012,2013
      Max Planck Institute for Polymer Research
  Copyright (C) 2008,2009,2010,2011
      Max-Planck-Institute for Polymer Research & Fraunhofer SCAI
  
  This file is part of ESPResSo++.
  
  ESPResSo++ is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  ESPResSo++ is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

// ESPP_CLASS


#include <cuda_runtime.h>
#include <stdio.h>
#include "LennardJonesGPU.cuh"
#include <math.h>
#define THREADSPERBLOCK 128

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) 
{
    unsigned long long int* address_as_ull =
                             (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__inline__ __device__
realG3 warpReduceSumTriple(realG3 val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val.x += __shfl_down_sync(0xFFFFFFFF, val.x, offset);
    val.y += __shfl_down_sync(0xFFFFFFFF, val.y, offset);
    val.z += __shfl_down_sync(0xFFFFFFFF, val.z, offset);
  }
  return val; 
}__inline__ __device__
realG warpReduceSum(realG val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}
__inline__ __device__
realG blockReduceSum(realG val) {

  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

  if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

  return val;
}
__inline__ __device__
realG3 blockReduceSumTriple(realG3 val) {

  static __shared__ realG3 shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSumTriple(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_realG3(0.0, 0.0, 0.0);

  if (wid==0) val = warpReduceSumTriple(val); //Final reduce within first warp

  return val;
}

using namespace std;
#define CUERR { \
  cudaError_t cudaerr; \
  if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
      printf("CUDA ERROR: \"%s\" in File %s at LINE %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__); \
  } \
}

#define PRINTL { \
  if(threadIdx.x == 0){ \
    printf("Line: %d\n", __LINE__); \
  } \
}


namespace espressopp {
  namespace interaction {

    __global__ void 
    testKernel( int nPart,
                int nCells,
                int* id,
                int* cellId,
                realG4* pos,
                realG4* force,
                realG* mass,
                realG* drift,
                int* type,
                bool* real,
                int* cellParticles, 
                int* cellOffsets,
                int* cellNeighbors,
                realG* energy,
                d_LennardJonesGPU* gpuPots,
                int numPots,
                int mode){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      
      extern __shared__ char parameter[];
      /*
      realG *s_cutoff = (realG*) &parameter[0];
      realG *s_sigma = (realG*) &parameter[sizeof(realG) * numPots];
      realG *s_epsilon = (realG*) &parameter[sizeof(realG) * 2 * numPots];
      realG *s_ff1 = (realG*) &parameter[sizeof(realG) * 3 * numPots];
      realG *s_ff2 = (realG*) &parameter[sizeof(realG) * 4 * numPots];
      
      __shared__ realG[10] cutoff;
      __shared__ realG[10] sigma;
      __shared__ realG[10] epsilon;
      __shared__ realG[10] ff1;
      __shared__ realG[10] ff2;
      __shared__ int calcMode;
      
      if(threadIdx.x == 0){
        for(int i = 0; i < numPots; ++i){
          s_cutoff[i] = gpuPots[i].cutoff;
          s_sigma[i] = gpuPots[i].sigma;
          s_epsilon[i] = gpuPots[i].epsilon;
          s_ff1[i] = gpuPots[i].ff1;
          s_ff2[i] = gpuPots[i].ff2;
        }
      }
      __syncthreads();
      */
      if(idx < nPart){
        realG3 p_pos;
        p_pos.x = pos[idx].x;
        p_pos.y = pos[idx].y;
        p_pos.z = pos[idx].z;
        //realG p_mass = mass[idx];
        //realG p_drift = drift[idx];
        int p_type = type[idx];
        //int p_real = real[idx] ? 1 : 0;
        int p_cellId = cellId[idx];
        realG3 p_force = make_realG3(0.0,0.0,0.0);
        realG3 p_dist;
        realG distSqr = 0;
        realG p_energy = 0;
        if(real[idx]){
          //#pragma unroll
          for(int i = 0; i < 27; ++i){
            int currentCellId = cellNeighbors[p_cellId * 27 + i];
            for(int j = 0; j < cellParticles[currentCellId]; ++j){
              int currentCellOffset = cellOffsets[currentCellId];
              if(currentCellOffset + j != idx){
                int potI = p_type * numPots + type[currentCellOffset + j];
                // p_dist.x = __dsub_rn(p_pos.x, pos[currentCellOffset + j].x);
                // p_dist.y = __dsub_rn(p_pos.y, pos[currentCellOffset + j].y);
                // p_dist.z = __dsub_rn(p_pos.z, pos[currentCellOffset + j].z);
                p_dist.x = p_pos.x - pos[currentCellOffset + j].x;
                p_dist.y = p_pos.y - pos[currentCellOffset + j].y;
                p_dist.z = p_pos.z - pos[currentCellOffset + j].z;
                distSqr =  p_dist.x * p_dist.x;
                distSqr += p_dist.y * p_dist.y;
                distSqr += p_dist.z * p_dist.z;
                // distSqr = 0;
                // distSqr = __fma_rn(p_dist.x, p_dist.x, distSqr);
                // distSqr = __fma_rn(p_dist.y, p_dist.y, distSqr);
                // distSqr = __fma_rn(p_dist.z, p_dist.z, distSqr);
                //if(distSqr <= (s_cutoff[potI] * s_cutoff[potI])){
                  if(distSqr <= (gpuPots[potI].cutoff * gpuPots[potI].cutoff)){
                    // if(distSqr <= __dmul_rn(gpuPots[potI].cutoff, gpuPots[potI].cutoff)){
                    if(mode == 0){
                      realG frac2 = 1.0 / distSqr;
                      // realG frac2 = __drcp_rn(distSqr);
                      realG frac6 = frac2 * frac2 * frac2;
                      // realG frac6 = __dmul_rn(frac2, __dmul_rn(frac2, frac2));
                      //realG ffactor = frac6 * (s_ff1[potI] * frac6 - s_ff2[potI]) * frac2;
                      realG ffactor = frac6 * (gpuPots[potI].ff1 * frac6 - gpuPots[potI].ff2) * frac2;
                      // realG ffactor = __dmul_rn(frac6, __dmul_rn((__dsub_rn(__dmul_rn(gpuPots[potI].ff1, frac6), gpuPots[potI].ff2)), frac2));
                      // p_force.x = __fma_rn(p_dist.x, ffactor, p_force.x);
                      // p_force.y = __fma_rn(p_dist.y, ffactor, p_force.y);
                      // p_force.z = __fma_rn(p_dist.z, ffactor, p_force.z);
                      p_force.x += p_dist.x * ffactor;
                      p_force.y += p_dist.y * ffactor;
                      p_force.z += p_dist.z * ffactor;
                    }
                    if(mode == 1){
                      //realG frac2 = s_sigma[potI] * s_sigma[potI] / distSqr;
                      realG frac2 = gpuPots[potI].sigma * gpuPots[potI].sigma / distSqr;
                      realG frac6 = frac2 * frac2 * frac2;
                      //realG energy = 4.0 * s_epsilon[potI] * (frac6 * frac6 - frac6);
                      realG f_energy = 4.0 * gpuPots[potI].epsilon * (frac6 * frac6 - frac6);
                      p_energy += f_energy;
                    }
                }
              }
            }
          }
        }
        if(mode == 0){
          force[idx].x = real[idx] * p_force.x;
          force[idx].y = real[idx] * p_force.y;
          force[idx].z = real[idx] * p_force.z;
        }
  
        if(mode == 1){
          energy[idx] = p_energy;
        }
      }
    }
    __global__ void 
    testKernel2( const int nPart,
                const int nCells,
                const int* id,
                const int* cellId,
                const realG4* pos,
                realG4* force,
                const realG* mass,
                const realG* drift,
                const int* type,
                const bool* real,
                const int* cellParticles, 
                const int* cellOffsets,
                const int* cellNeighbors,
                realG* energy,
                const d_LennardJonesGPU* gpuPots,
                const int numPots,
                const int mode){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      // __shared__ realG4 s_pos[THREADSPERBLOCK];
      __shared__ realG s_pos_x[THREADSPERBLOCK];
      __shared__ realG s_pos_y[THREADSPERBLOCK];
      __shared__ realG s_pos_z[THREADSPERBLOCK];
      __shared__ int s_id[THREADSPERBLOCK];
      //__shared__ realG[THREADSPERBLOCK] s_mass;
      //__shared__ realG[THREADSPERBLOCK] s_drift;
      __shared__ int s_type[THREADSPERBLOCK];
      __shared__ int activeThreads;
      // __shared__ realG s_force_x[THREADSPERBLOCK];
      // __shared__ realG s_force_y[THREADSPERBLOCK];
      // __shared__ realG s_force_z[THREADSPERBLOCK];
      __shared__ realG s_energy[THREADSPERBLOCK];
      __shared__ int numberRuns;
      int potI;
      bool sameId;
      int calcCellOffset = cellOffsets[blockIdx.x];
      // s_force_x[threadIdx.x] = 0.0f;
      // s_force_y[threadIdx.x] = 0.0f;
      // s_force_z[threadIdx.x] = 0.0f;
      realG p_energy;

      s_energy[threadIdx.x] = 0.0f;
      p_energy = 0.0f;

      if(cellParticles[blockIdx.x] == 0){
        return;
      }
      if(real[calcCellOffset] == false){
        return;
      }
      if(threadIdx.x == 0){
        int numberNeighborParticles = 0;
        for(int i = 0; i < 27; ++i){
            numberNeighborParticles += cellParticles[cellNeighbors[blockIdx.x * 27 + i]];
        }
        numberRuns = (numberNeighborParticles - 1) / THREADSPERBLOCK + 1;
      }

      int currentii = 0;
      int currentjj = 0;

      __syncthreads(); 

      realG3 p_dist;
      realG3 p_force;
      realG distSqr;
      for(int i = 0; i < numberRuns; ++i){
        activeThreads = 0;
        int ii,jj;
        if(threadIdx.x == 0){
          for(ii = currentii; ii < 27; ++ii){
            // if(blockIdx.x == 41 && mode == 0) {
            //   printf("accessing neighbor cell: %d\n", cellNeighbors[blockIdx.x * 27 + ii]);
            // }
            for(jj = currentjj; jj < cellParticles[cellNeighbors[blockIdx.x * 27 + ii]]; ++jj){
              if(activeThreads == THREADSPERBLOCK){ //} || (ii == 26 && jj == cellParticles[cellNeighbors[blockIdx.x * 27 + 26]] - 1)){
                //sharedMemfull = true;
                currentii = ii;
                currentjj = jj;
                goto end;
                // if(jj == cellParticles[cellNeighbors[blockIdx.x * 27 + ii]] - 1){
                //   currentii = ii+1;
                //   currentjj = 0;
                // } else {
                //   currentjj = jj + 1;
                //   currentii = ii;
                // }
              } else{
                // if(blockIdx.x == 41 && mode == 0) {
                //   printf("activeThreads %d, ii %d, jj %d, blockIdx %d\n", activeThreads, ii, jj, blockIdx.x);
                // }
                s_pos_x[activeThreads] = pos[cellOffsets[cellNeighbors[blockIdx.x * 27 + ii]] + jj].x;
                s_pos_y[activeThreads] = pos[cellOffsets[cellNeighbors[blockIdx.x * 27 + ii]] + jj].y;
                s_pos_z[activeThreads] = pos[cellOffsets[cellNeighbors[blockIdx.x * 27 + ii]] + jj].z;
                s_type[activeThreads] = type[cellOffsets[cellNeighbors[blockIdx.x * 27 + ii]] + jj];
                s_id[activeThreads] = id[cellOffsets[cellNeighbors[blockIdx.x * 27 + ii]] + jj];
                activeThreads++;
              }
            }
            currentjj = 0;
          }
          end:;
        }
        __syncthreads();
        for(int j = 0; j < cellParticles[blockIdx.x]; ++j){
          // s_force_x[threadIdx.x] = 0.0f;
          // s_force_y[threadIdx.x] = 0.0f;
          // s_force_z[threadIdx.x] = 0.0f;
          s_energy[threadIdx.x] = 0.0f;
          p_energy = 0.0f;
          p_force.x = 0.0f;
          p_force.y = 0.0f;
          p_force.z = 0.0f;
          potI = s_type[threadIdx.x] * numPots + type[calcCellOffset + j];
          sameId = s_id[threadIdx.x] == id[calcCellOffset + j] ? true : false;
          if(threadIdx.x < activeThreads){
            //printf("threadIdx.x=%d, idx.x: %d, own particle id: %d\n", threadIdx.x, idx, s_id[threadIdx.x]);
            // p_dist.x = s_pos[threadIdx.x].x - pos[calcCellOffset + j].x;
            // p_dist.y = s_pos[threadIdx.x].y - pos[calcCellOffset + j].y;
            // p_dist.z = s_pos[threadIdx.x].z - pos[calcCellOffset + j].z;
            p_dist.x = pos[calcCellOffset + j].x - s_pos_x[threadIdx.x];
            p_dist.y = pos[calcCellOffset + j].y - s_pos_y[threadIdx.x];
            p_dist.z = pos[calcCellOffset + j].z - s_pos_z[threadIdx.x];
            distSqr =  p_dist.x * p_dist.x + p_dist.y * p_dist.y + p_dist.z * p_dist.z;
            if(distSqr <= (gpuPots[potI].cutoff * gpuPots[potI].cutoff)){
              if(!sameId){
                if(mode == 0){
                  realG frac2 = 1.0 / distSqr;
                  realG frac6 = frac2 * frac2 * frac2;
                  realG ffactor = frac6 * (gpuPots[potI].ff1 * frac6 - gpuPots[potI].ff2) * frac2;
                  // s_force_x[threadIdx.x] = p_dist.x * ffactor;
                  // s_force_y[threadIdx.x] = p_dist.y * ffactor;
                  // s_force_z[threadIdx.x] = p_dist.z * ffactor;
                  p_force.x = p_dist.x * ffactor;
                  p_force.y = p_dist.y * ffactor;
                  p_force.z = p_dist.z * ffactor;
                  //printf("2. id1=%d, id2=%d %f %f %f\n", id[calcCellOffset + j], s_id[threadIdx.x], p_dist.x * ffactor,  p_dist.y * ffactor, p_dist.z * ffactor);

                }
                if(mode == 1){
                  realG frac2 = gpuPots[potI].sigma * gpuPots[potI].sigma / distSqr;
                  realG frac6 = frac2 * frac2 * frac2;
                  realG energy = 4.0 * gpuPots[potI].epsilon * (frac6 * frac6 - frac6);
                  p_energy = energy;
                  //printf("Energy threadIdx: %d, %f\n",threadIdx.x, p_energy);
                  s_energy[threadIdx.x] = energy;
                }
              }
            }
          }
          __syncthreads();
          if(mode == 0){
            p_force = blockReduceSumTriple(p_force);
            // if(threadIdx.x == 0){
            //   for(int k = 0; k < activeThreads; ++k){
            //     force[calcCellOffset + j].x += s_force_x[k];
            //     force[calcCellOffset + j].y += s_force_y[k];
            //     force[calcCellOffset + j].z += s_force_z[k];
            //   }
            // }
            // if(threadIdx.x < activeThreads){
            //   force[calcCellOffset + j].x += s_force_x[threadIdx.x];
            //   force[calcCellOffset + j].y += s_force_y[threadIdx.x];
            //   force[calcCellOffset + j].z += s_force_z[threadIdx.x];
            if(threadIdx.x == 0){
              force[calcCellOffset + j].x += p_force.x;
              force[calcCellOffset + j].y += p_force.y;
              force[calcCellOffset + j].z += p_force.z;
            }
          }
          if(mode == 1){
            p_energy = blockReduceSum(p_energy);
            if(threadIdx.x == 0){
              // energy[calcCellOffset + j] += p_energy;
              for(int k = 0; k < activeThreads; ++k){
                energy[calcCellOffset + j] += s_energy[k];
              }
              //printf("pos in memory:: %d, Energy blockIdx: %d, Penergy: %f\n", calcCellOffset + j, blockIdx.x, energy[calcCellOffset + j]);
            }
          }
          __syncthreads();
        }
        __syncthreads();
      }
    }

    __global__ void 
    testKernel3( const int nPart,
                const int nCells,
                const int* id,
                const int* cellId,
                const realG4* pos,
                realG4* force,
                const realG* mass,
                const realG* drift,
                const int* type,
                const bool* real,
                const int* cellParticles, 
                const int* cellOffsets,
                const int* cellNeighbors,
                realG* energy,
                const d_LennardJonesGPU* gpuPots,
                const int numPots,
                const int mode){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      __shared__ int numberLineParticles[9];
      __shared__ int numberLineWarps[9];

      int potI;
      bool sameId;
      int calcCellOffset = cellOffsets[blockIdx.x];
      

      if(cellParticles[blockIdx.x] == 0){
        return;
      }
      if(real[calcCellOffset] == false){
        return;
      }

      __syncthreads(); 

      realG3 p_dist;
      realG3 p_force;
      realG distSqr;
      realG t_pos_x;
      realG t_pos_y;
      realG t_pos_z;      
      realG t_type;
      realG t_id;
      realG p_energy = 0;

      int warpId = threadIdx.x / warpSize;
      int laneId = threadIdx.x % warpSize;
      if(warpId >= 9){
        printf("error");
      }
      int dataOffset = cellNeighbors[blockIdx.x * 27 + 3 * warpId];

      if(threadIdx.x < 9){
        numberLineParticles[threadIdx.x] = cellParticles[cellNeighbors[(blockIdx.x * 27) + (3 * threadIdx.x)] + 0] +
                                            cellParticles[cellNeighbors[(blockIdx.x * 27) + (3 * threadIdx.x)] + 1] +
                                            cellParticles[cellNeighbors[(blockIdx.x * 27) + (3 * threadIdx.x)] + 2];
        numberLineWarps[threadIdx.x] = (numberLineParticles[threadIdx.x ] - 1) / warpSize + 1;
      }
      __syncthreads();

      for(int i = 0; i < numberLineWarps[warpId]; ++i){
        t_pos_x = 0.0f;
        t_pos_y = 0.0f;
        t_pos_z = 0.0f;
        if(i * warpSize + laneId < numberLineParticles[warpId]){
          t_pos_x = pos[dataOffset + i * warpSize + laneId].x;
          t_pos_y = pos[dataOffset + i * warpSize + laneId].y;
          t_pos_z = pos[dataOffset + i * warpSize + laneId].z;
          t_type = type[dataOffset + i * warpSize + laneId];
          t_id = id[dataOffset + i * warpSize + laneId];
      
          for(int j = 0; j < cellParticles[blockIdx.x]; ++j){
            p_energy = 0.0f;
            p_force.x = 0.0f;
            p_force.y = 0.0f;
            p_force.z = 0.0f;
            potI = t_id * numPots + type[calcCellOffset + j];
            sameId = t_id == id[calcCellOffset + j] ? true : false;

            p_dist.x = pos[calcCellOffset + j].x - t_pos_x;
            p_dist.y = pos[calcCellOffset + j].y - t_pos_y;
            p_dist.z = pos[calcCellOffset + j].z - t_pos_z;
            distSqr =  p_dist.x * p_dist.x + p_dist.y * p_dist.y + p_dist.z * p_dist.z;
            if(distSqr <= (gpuPots[potI].cutoff * gpuPots[potI].cutoff)){
              if(!sameId){
                if(mode == 0){
                  realG frac2 = 1.0 / distSqr;
                  realG frac6 = frac2 * frac2 * frac2;
                  realG ffactor = frac6 * (gpuPots[potI].ff1 * frac6 - gpuPots[potI].ff2) * frac2;
                  p_force.x = p_dist.x * ffactor;
                  p_force.y = p_dist.y * ffactor;
                  p_force.z = p_dist.z * ffactor;
                }
                if(mode == 1){
                  realG frac2 = gpuPots[potI].sigma * gpuPots[potI].sigma / distSqr;
                  realG frac6 = frac2 * frac2 * frac2;
                  realG energy = 4.0 * gpuPots[potI].epsilon * (frac6 * frac6 - frac6);
                  p_energy = energy;
                }
              }
            }
            
            __syncwarp();
            if(mode == 0){
              p_force = blockReduceSumTriple(p_force);
              if(threadIdx.x == 0){
                atomicAdd(&force[calcCellOffset + j].x, p_force.x);
                atomicAdd(&force[calcCellOffset + j].y, p_force.y);
                atomicAdd(&force[calcCellOffset + j].z, p_force.z);
                // force[calcCellOffset + j].x += p_force.x;
                // force[calcCellOffset + j].y += p_force.y;
                // force[calcCellOffset + j].z += p_force.z;
              }
            }
            if(mode == 1){
              p_energy = blockReduceSum(p_energy);
              if(threadIdx.x == 0){
                atomicAdd(&energy[calcCellOffset + j], p_energy);
                // energy[calcCellOffset + j] += p_energy;
              }
            }
          }
        }
      }
    }
    
  __global__ void
  tKern(int N, realG4* force){
    float totalForce = 0;
    for(int i=0; i<N; ++i){
      totalForce += fabs(force[i].x) + fabs(force[i].y) + fabs(force[i].z); 
      // if(force[i].x != 0.0f){
      //   printf("Force[%4d] xyz: %.8f      %.8f      %.8f\n", i, force[i].x, force[i].y, force[i].z);
      // }
    }
    printf("%f \t TotalForce\n", totalForce);
  }

  realG LJGPUdriver(StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots, int mode){
    int numThreads = 128;
    int numBlocks = (gpuStorage->numberLocalParticles) / numThreads + 1;
    realG *h_energy; 
    realG *d_energy;
    realG totalEnergy = 0;

    h_energy = new realG[gpuStorage->numberLocalParticles];
    cudaMalloc(&d_energy, sizeof(realG) * gpuStorage->numberLocalParticles);
    cudaMemset(d_energy, 0, sizeof(realG) * gpuStorage->numberLocalParticles);
    unsigned numPots = 1;
    unsigned shared_mem_size = 10 * sizeof(realG) * 5;
    cudaMemset(gpuStorage->d_force, 0, sizeof(realG4) * gpuStorage->numberLocalParticles);
    if(false){
      testKernel<<<numBlocks, numThreads, shared_mem_size>>>(
                              gpuStorage->numberLocalParticles, 
                              gpuStorage->numberLocalCells, 
                              gpuStorage->d_id,
                              gpuStorage->d_cellId,
                              gpuStorage->d_pos,
                              gpuStorage->d_force,
                              gpuStorage->d_mass,
                              gpuStorage->d_drift,
                              gpuStorage->d_type,
                              gpuStorage->d_real,
                              gpuStorage->d_particlesCell,
                              gpuStorage->d_cellOffsets,
                              gpuStorage->d_cellNeighbors,
                              d_energy,
                              gpuPots,
                              numPots,
                              mode
                            );
    } else{
      testKernel3<<<gpuStorage->numberLocalCells, THREADSPERBLOCK>>>(
        gpuStorage->numberLocalParticles, 
        gpuStorage->numberLocalCells, 
        gpuStorage->d_id,
        gpuStorage->d_cellId,
        gpuStorage->d_pos,
        gpuStorage->d_force,
        gpuStorage->d_mass,
        gpuStorage->d_drift,
        gpuStorage->d_type,
        gpuStorage->d_real,
        gpuStorage->d_particlesCell,
        gpuStorage->d_cellOffsets,
        gpuStorage->d_cellNeighbors,
        d_energy,
        gpuPots,
        numPots,
        mode
      ); CUERR
    }
    //cudaDeviceSynchronize();

      //printf("---\n");
      if(mode == 1) {
        cudaMemcpy(h_energy, d_energy, sizeof(realG) * gpuStorage->numberLocalParticles, cudaMemcpyDeviceToHost); CUERR
        for (int i = 0; i < gpuStorage->numberLocalParticles; ++i){ 
          totalEnergy += h_energy[i];
        }
      }

      return totalEnergy / (double)2.0f;
    }
  }
}

