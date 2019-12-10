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
#include <assert.h>
#define THREADSPERBLOCK 1024
//#ifdef __NVCC__

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

template<typename T>
__device__ __forceinline__ T ldg(const T* ptr) {
return *ptr;
}


namespace espressopp {
  namespace interaction {

__global__ void 
    verletListKernel( const int nPart,
                const realG3* __restrict__ pos,
                realG3* force,
                const realG* __restrict__ mass,
                const realG* __restrict__ drift,
                const int* __restrict__ type,
                const bool* __restrict__ real,
                realG* energy,
                const d_LennardJonesGPU* __restrict__ gpuPots,
                int numPots,
                int mode,
                const int* __restrict__ vl,
                const int* __restrict__ num_nb){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      extern __shared__ char parameter[];
      
      realG *s_cutoff = (realG*) &parameter[0];
      realG *s_sigma = (realG*) &parameter[sizeof(realG) * numPots];
      realG *s_epsilon = (realG*) &parameter[sizeof(realG) * 2 * numPots];
      realG *s_ff1 = (realG*) &parameter[sizeof(realG) * 3 * numPots];
      realG *s_ff2 = (realG*) &parameter[sizeof(realG) * 4 * numPots];
            
      if(threadIdx.x < numPots){
          s_cutoff[threadIdx.x] = gpuPots[threadIdx.x].cutoff;
          s_sigma[threadIdx.x] = gpuPots[threadIdx.x].sigma;
          s_epsilon[threadIdx.x] = gpuPots[threadIdx.x].epsilon;
          s_ff1[threadIdx.x] = gpuPots[threadIdx.x].ff1;
          s_ff2[threadIdx.x] = gpuPots[threadIdx.x].ff2;
      }
      __syncthreads();
      
      
      if (idx >= nPart) return;
      if (!real[idx]) return;

      realG3 p_pos = pos[idx];
      int p_type = type[idx];
      //realG p_mass = mass[idx];
      //realG p_drift = drift[idx];
      realG3 p_force = make_realG3(0.0,0.0,0.0);
      realG p_energy = 0;
      int n_nb = num_nb[idx];

      for(int i = 0; i < n_nb; ++i){
        int p2_idx = vl[i * nPart + idx];

        realG3 p_dist;
        realG3 p2_pos = pos[p2_idx];
        int potIdx = p_type * numPots + type[p2_idx];
        p_dist = p_pos - p2_pos;
        realG distSqr = dot(p_dist, p_dist);
        if(distSqr <= (s_cutoff[potIdx] * s_cutoff[potIdx])){
          if(mode == 0){
            realG frac2 = 1.0 / distSqr;
            realG frac6 = frac2 * frac2 * frac2;
            realG calcResult = frac6 * (s_ff1[potIdx] * frac6 - s_ff2[potIdx]) * frac2;
            p_force += p_dist * calcResult;
          }
          if(mode == 1){
            realG frac2 = s_sigma[potIdx] * s_sigma[potIdx] / distSqr;
            realG frac6 = frac2 * frac2 * frac2;
            realG calcResult = 4.0 * s_epsilon[potIdx] * (frac6 * frac6 - frac6);
            p_energy += calcResult;
          }
        }
      }

      if(mode == 0){
        force[idx] = p_force;
      }

      if(mode == 1){
        energy[idx] = p_energy;
      }
    }
    __global__ void 
    testKernel( const int nPart,
                const int nCells,
                const int* __restrict__ id,
                const int* __restrict__ cellId,
                const realG3* __restrict__ pos,
                realG3* force,
                const realG* __restrict__ mass,
                const realG* __restrict__ drift,
                const int* __restrict__ type,
                const bool* __restrict__ real,
                const int* __restrict__ cellParticlesN, 
                const int* __restrict__ cellOffsets,
                const int* __restrict__ cellNeighbors,
                realG* energy,
                const d_LennardJonesGPU* __restrict__ gpuPots,
                int numPots,
                int mode){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      extern __shared__ char parameter[];
      
      realG *s_cutoff = (realG*) &parameter[0];
      realG *s_sigma = (realG*) &parameter[sizeof(realG) * numPots];
      realG *s_epsilon = (realG*) &parameter[sizeof(realG) * 2 * numPots];
      realG *s_ff1 = (realG*) &parameter[sizeof(realG) * 3 * numPots];
      realG *s_ff2 = (realG*) &parameter[sizeof(realG) * 4 * numPots];
            
      if(threadIdx.x < numPots){
          s_cutoff[threadIdx.x] = gpuPots[threadIdx.x].cutoff;
          s_sigma[threadIdx.x] = gpuPots[threadIdx.x].sigma;
          s_epsilon[threadIdx.x] = gpuPots[threadIdx.x].epsilon;
          s_ff1[threadIdx.x] = gpuPots[threadIdx.x].ff1;
          s_ff2[threadIdx.x] = gpuPots[threadIdx.x].ff2;
      }

      if (idx >= nPart) return;
      if (!real[idx]) return;

      // __shared__ realG3 s_pos1[THREADSPERBLOCK];
      //__shared__ realG3 s_pos2[THREADSPERBLOCK];
      // __shared__ int s_type[THREADSPERBLOCK];
      // __shared__ int s_cellId[THREADSPERBLOCK];
      // __shared__ realG3 s_force[THREADSPERBLOCK];
      // __shared__ realG s_energy[THREADSPERBLOCK];
      __syncthreads();
      
      


      realG3 p_pos = pos[idx];
      // s_pos1[threadIdx.x] = pos[idx];
      //realG p_mass = mass[idx];
      //realG p_drift = drift[idx];
      int p_type = type[idx];
      // s_type[threadIdx.x] = ldg(type+idx);
      int p_cellId = cellId[idx];
      // s_cellId[threadIdx.x] = ldg(cellId+idx);
      realG3 p_force = make_realG3(0.0,0.0,0.0);
      // s_force[threadIdx.x] =  make_realG3(0.0,0.0,0.0);
      realG p_energy = 0;
      // s_energy[threadIdx.x] = 0;
      // #pragma unroll
      for(int i = 0; i < 27; ++i){
        int currentCellId = *(cellNeighbors+(p_cellId * 27 + i));
        // int currentCellId = ldg(cellNeighbors+(s_cellId[threadIdx.x] * 27 + i));
        int sizeCell = *(cellParticlesN+currentCellId);
        int currentCellOffset = *(cellOffsets+currentCellId);
        for(int j = 0; j < sizeCell; ++j){
          int pOffset = currentCellOffset + j;
          if(pOffset != idx){
            int potIdx = p_type * numPots + *(type+pOffset);
            // int potIdx = s_type[threadIdx.x] * numPots + ldg(type+pOffset);
            realG3 p2_pos = pos[pOffset];
            // s_pos2[idx] = pos[pOffset];         
            realG3 distVec = p_pos - p2_pos;
            realG distSqr = dot(distVec, distVec);
            // distVec.x * distVec.x + distVec.y * distVec.y + distVec.z * distVec.z;
            if(distSqr <= (s_cutoff[potIdx] * s_cutoff[potIdx])){
            // if(distSqr <= (gpuPots[potIdx].cutoff * gpuPots[potIdx].cutoff)){
              if(mode == 0){
                realG frac2 = 1.0 / distSqr;
                realG frac6 = frac2 * frac2 * frac2;
                realG calcResult = frac6 * (s_ff1[potIdx] * frac6 - s_ff2[potIdx]) * frac2;
                // realG calcResult = frac6 * (gpuPots[potIdx].ff1 * frac6 - gpuPots[potIdx].ff2) * frac2;
                p_force += distVec * calcResult;
                // s_force[threadIdx.x] += distVec * calcResult;
              }
              if(mode == 1){
                realG frac2 = s_sigma[potIdx] * s_sigma[potIdx] / distSqr;
                // realG frac2 = gpuPots[potIdx].sigma * gpuPots[potIdx].sigma / distSqr;
                realG frac6 = frac2 * frac2 * frac2;
                realG calcResult = 4.0 * s_epsilon[potIdx] * (frac6 * frac6 - frac6);
                // realG calcResult = 4.0 * gpuPots[potIdx].epsilon * (frac6 * frac6 - frac6);
                p_energy += calcResult;
                // s_energy[threadIdx.x] += calcResult;
              }
            }
          }
          // __syncwarp();
        }
        // __syncwarp();
      }
      if(mode == 0){
        force[idx] = p_force;
        // force[idx] = s_force[threadIdx.x];
      }

      if(mode == 1){
        energy[idx] = p_energy;
        // energy[idx] = s_energy[threadIdx.x];
      }
    }
    __global__ void 
    testKernel2( const int nPart,
                const int nCells,
                const int* id,
                const int* cellId,
                const realG3* pos,
                realG3* force,
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
      // __shared__ realG3 s_pos[THREADSPERBLOCK];
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
            p_force = blockReduceSumTriple(p_force, 0xffffffff);
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
            p_energy = blockReduceSum(p_energy, 0xffffffff);
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
    testKernel3(const int nPart,
                const int nCells,
                const int* __restrict__ id,
                const int* __restrict__ cellId,
                const realG3* __restrict__ pos,
                realG3* force,
                const realG* __restrict__ mass,
                const realG* __restrict__ drift,
                const int* __restrict__ type,
                const bool* __restrict__ real,
                const int* __restrict__ cellParticles, 
                const int* __restrict__ cellOffsets,
                const int* __restrict__ cellNeighbors,
                realG* __restrict__ energy,
                const d_LennardJonesGPU* __restrict__ gpuPots,
                int numPots,
                int mode){
      // int idx = blockIdx.x * blockDim.x + threadIdx.x;
      __shared__ int numberLineParticles[9];
      __shared__ int numberLineWarps[9];

      int calcCellOffset = cellOffsets[blockIdx.x];

      if(cellParticles[blockIdx.x] == 0){
        return;
      }
      if(real[calcCellOffset] == false){
        return;
      }


      int warpId = threadIdx.x / warpSize;
      int laneId = threadIdx.x % warpSize;

      int dataOffset = cellOffsets[cellNeighbors[blockIdx.x * 27 + 3 * warpId]];

      if(threadIdx.x < 9){
        numberLineParticles[threadIdx.x] = cellParticles[cellNeighbors[(blockIdx.x * 27) + (3 * threadIdx.x) + 0]] +
                                            cellParticles[cellNeighbors[(blockIdx.x * 27) + (3 * threadIdx.x) + 1]] +
                                            cellParticles[cellNeighbors[(blockIdx.x * 27) + (3 * threadIdx.x) + 2]];
        numberLineWarps[threadIdx.x] = (numberLineParticles[threadIdx.x ] - 1) / warpSize + 1;
      }
      __syncthreads();

      for(int i = 0; i < numberLineWarps[warpId]; ++i){
        unsigned mask = __ballot_sync(0xffffffff, i * warpSize + laneId < numberLineParticles[warpId]);
        if(i * warpSize + laneId < numberLineParticles[warpId]){
          realG3 t_pos = pos[dataOffset + i * warpSize + laneId];
          int t_type = type[dataOffset + i * warpSize + laneId];
          int t_id = id[dataOffset + i * warpSize + laneId];
          for(int j = 0; j < cellParticles[blockIdx.x]; ++j){
            realG3 p_force;
            realG p_energy;
            p_energy = 0.0f;
            p_force.x = 0.0f;
            p_force.y = 0.0f;
            p_force.z = 0.0f;
            int potI = t_type * numPots + type[calcCellOffset + j];
            bool sameId = t_id == id[calcCellOffset + j] ? true : false;

            realG3 p_dist;
            p_dist = pos[calcCellOffset + j] - t_pos;

            realG distSqr =  dot(p_dist, p_dist);
            if(distSqr <= (gpuPots[potI].cutoff * gpuPots[potI].cutoff)){
              if(!sameId){
                if(mode == 0){
                  realG frac2 = 1.0 / distSqr;
                  realG frac6 = frac2 * frac2 * frac2;
                  realG ffactor = frac6 * (gpuPots[potI].ff1 * frac6 - gpuPots[potI].ff2) * frac2;
                  p_force = p_dist * ffactor;
                }
                if(mode == 1){
                  realG frac2 = gpuPots[potI].sigma * gpuPots[potI].sigma / distSqr;
                  realG frac6 = frac2 * frac2 * frac2;
                  p_energy = 4.0 * gpuPots[potI].epsilon * (frac6 * frac6 - frac6);
                }
              }
            }
          
            // __syncthreads();
            //__syncwarp();
            if(mode == 0){
              p_force = warpReduceSumTriple(p_force, mask);
              if(laneId == 0){
                atomicAdd(&force[calcCellOffset + j].x, p_force.x);
                atomicAdd(&force[calcCellOffset + j].y, p_force.y);
                atomicAdd(&force[calcCellOffset + j].z, p_force.z);
              }
            }
            if(mode == 1){
              p_energy = warpReduceSum(p_energy, mask);
              if(laneId == 0){
                  atomicAdd(&energy[calcCellOffset + j], p_energy);
              }
            }
            __syncwarp();
          }
        }
      }
    }
    
  realG LJGPUdriverVl(StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots, int ptypes, int* vl, int* n_nb, int mode){
    realG *h_energy; 
    realG *d_energy;
    realG totalEnergy = 0;

    if(mode == 1) {
      h_energy = new realG[gpuStorage->numberLocalParticles];
      cudaMalloc(&d_energy, sizeof(realG) * gpuStorage->numberLocalParticles); CUERR
      cudaMemset(d_energy, 0, sizeof(realG) * gpuStorage->numberLocalParticles); CUERR
      cudaMemset(gpuStorage->d_force, 0, sizeof(realG3) * gpuStorage->numberLocalParticles); CUERR
    }
    
    unsigned shared_mem_size = ptypes * ptypes * sizeof(realG) * 5;
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start); CUERR
    // cudaEventCreate(&stop); CUERR
    // cudaEventRecord(start); CUERR

    verletListKernel<<<SDIV(gpuStorage->numberLocalParticles, THREADSPERBLOCK), THREADSPERBLOCK, shared_mem_size>>>(
      gpuStorage->numberLocalParticles, 
      gpuStorage->d_pos,
      gpuStorage->d_force,
      gpuStorage->d_mass,
      gpuStorage->d_drift,
      gpuStorage->d_type,
      gpuStorage->d_real,
      d_energy,
      gpuPots,
      ptypes,
      mode,
      vl,
      n_nb
    );

    // cudaDeviceSynchronize(); CUERR
    // cudaEventRecord(stop); CUERR
    // cudaEventSynchronize(stop); CUERR
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop); CUERR
    // printf("%s kernel time: %2.6f\n", mode==0? "Force" : "Energy", milliseconds);
      if(mode == 1) {
        cudaMemcpy(h_energy, d_energy, sizeof(realG) * gpuStorage->numberLocalParticles, cudaMemcpyDeviceToHost); CUERR
        for (int i = 0; i < gpuStorage->numberLocalParticles; ++i){ 
          totalEnergy += h_energy[i];
        }
        cudaFree(d_energy);
      }
      return totalEnergy / (double)2.0;

  }

  realG LJGPUdriver(StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots, int ptypes, int mode){
    realG totalEnergy = 0;
    realG *h_energy; 
    realG *d_energy;
    if(mode == 1) {      
      h_energy = new realG[gpuStorage->numberLocalParticles];
      cudaMalloc(&d_energy, sizeof(realG) * gpuStorage->numberLocalParticles); CUERR
      cudaMemset(d_energy, 0, sizeof(realG) * gpuStorage->numberLocalParticles); CUERR
    }
    // cudaMemset(gpuStorage->d_force, 0, sizeof(realG3) * gpuStorage->numberLocalParticles); CUERR
    unsigned shared_mem_size = ptypes * sizeof(realG) * 5;
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start); CUERR
    // cudaEventCreate(&stop); CUERR
    // cudaEventRecord(start); CUERR
    testKernel<<<SDIV(gpuStorage->numberLocalParticles, THREADSPERBLOCK), THREADSPERBLOCK, shared_mem_size>>>(
      //  testKernel2<<<gpuStorage->numberLocalCells, THREADSPERBLOCK>>>(
      // testKernel3<<<gpuStorage->numberLocalCells, 288>>>(
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
                            ptypes,
                            mode
                          );
  
    // cudaDeviceSynchronize(); CUERR
    // cudaEventRecord(stop); CUERR
    // cudaEventSynchronize(stop); CUERR
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop); CUERR
    // printf("%s kernel time: %2.6f\n", mode==0? "Force" : "Energy", milliseconds);
    if(mode == 1) {
      cudaMemcpy(h_energy, d_energy, sizeof(realG) * gpuStorage->numberLocalParticles, cudaMemcpyDeviceToHost); CUERR
      for (int i = 0; i < gpuStorage->numberLocalParticles; ++i){ 
        totalEnergy += h_energy[i];
      }
      cudaFree(d_energy);
    }
    return totalEnergy / (double)2.0;
    }
  }
}

// #endif