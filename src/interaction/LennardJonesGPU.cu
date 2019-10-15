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


using namespace std;
#define CUERR { \
  cudaError_t cudaerr; \
  if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
      printf("CUDA ERROR: \"%s\" in File %s at LINE %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__); \
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
      realG *s_cutoff = (realG*) &parameter[0];
      realG *s_sigma = (realG*) &parameter[sizeof(realG) * numPots];
      realG *s_epsilon = (realG*) &parameter[sizeof(realG) * 2 * numPots];
      realG *s_ff1 = (realG*) &parameter[sizeof(realG) * 3 * numPots];
      realG *s_ff2 = (realG*) &parameter[sizeof(realG) * 4 * numPots];
      /*
      __shared__ realG[10] cutoff;
      __shared__ realG[10] sigma;
      __shared__ realG[10] epsilon;
      __shared__ realG[10] ff1;
      __shared__ realG[10] ff2;
      __shared__ int calcMode;
      */
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
        realG distSqr;
        realG p_energy = 0;
        if(real[idx]){
          for(int i = 0; i < 27; ++i){
            int currentCellId = cellNeighbors[p_cellId * 27 + i];
            for(int j = 0; j < cellParticles[currentCellId]; ++j){
              int currentCellOffset = cellOffsets[currentCellId];
              if(currentCellOffset + j != idx){
                int potI = p_type * numPots + type[currentCellOffset + j];
                p_dist.x = p_pos.x - pos[currentCellOffset + j].x;
                p_dist.y = p_pos.y - pos[currentCellOffset + j].y;
                p_dist.z = p_pos.z - pos[currentCellOffset + j].z;
                distSqr =  p_dist.x * p_dist.x;
                distSqr += p_dist.y * p_dist.y;
                distSqr += p_dist.z * p_dist.z;
                if(distSqr <= (s_cutoff[potI] * s_cutoff[potI])){
                  if(mode == 0){
                    realG frac2 = 1.0 / distSqr;
                    realG frac6 = frac2 * frac2 * frac2;
                    realG ffactor = frac6 * (s_ff1[potI] * frac6 - s_ff2[potI]) * frac2;
                    p_force.x += p_dist.x * ffactor;
                    p_force.y += p_dist.y * ffactor;
                    p_force.z += p_dist.z * ffactor;
                  }
                  if(mode == 1){
                    realG frac2 = s_sigma[potI] * s_sigma[potI] / distSqr;
                    realG frac6 = frac2 * frac2 * frac2;
                    realG energy = 4.0 * s_epsilon[potI] * (frac6 * frac6 - frac6);
                    p_energy += energy;
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

    

  realG LJGPUdriver(StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots, int mode){
    int numThreads = 128;
    int numBlocks = (gpuStorage->numberLocalParticles) / numThreads + 1;
    realG *h_energy; 
    realG *d_energy;
    realG totalEnergy = 0;

    h_energy = new realG[gpuStorage->numberLocalParticles];
    cudaMalloc(&d_energy, sizeof(realG) * gpuStorage->numberLocalParticles);
    unsigned numPots = 1;
    unsigned shared_mem_size = 10 * sizeof(realG) * 5;
    //printf("numLocalCells: %d, Particles: %d, numBlocks: %d, numThreads: %d\n", gpuStorage->numberLocalCells, gpuStorage->numberLocalParticles, numBlocks, numBlocks*numThreads);
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
      cudaDeviceSynchronize();  CUERR

      //printf("---\n");
      if(mode == 1) {
        cudaMemcpy(h_energy, d_energy, sizeof(realG) * gpuStorage->numberLocalParticles, cudaMemcpyDeviceToHost);
        for (int i = 0; i < gpuStorage->numberLocalParticles; ++i){
          totalEnergy += h_energy[i];
        }
      }
      return totalEnergy / 2;
    }
  }
}

