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
                double3* pos,
                double3* force,
                double* mass,
                double* drift,
                int* type,
                bool* real,
                int* cellParticles, 
                int* cellOffsets,
                int* cellNeighbors,
                double* energy,
                d_LennardJonesGPU* gpuPots,
                int mode){
      int idx = blockIdx.x * blockDim.x + threadIdx.x;

      __shared__ double cutoff;
      __shared__ int calcMode;
      //__shared__ d_LennardJonesGPU potential;

      if(threadIdx.x == 0){
        cutoff = gpuPots[0].getCutoff();
        calcMode = mode;
        //potential = gpuPots[0];
      }
      __syncthreads();

      

      if(idx < nPart){
        double3 p_pos = pos[idx];
        double p_mass = mass[idx];
        double p_drift = drift[idx];
        int p_type = type[idx];
        int p_real = real[idx] ? 1 : 0;
        int p_cellId = cellId[idx];
        //double3 p_force = make_double3(0.0,0.0,0.0);
        double3 test_force = make_double3(0.0,0.0,0.0);
        double3 p_dist;
        double distSqr;
        double p_energy = 0;
        if(p_real == 1){
          ///*
          for(int i = 0; i < 27; i++){
            int currentCellId = cellNeighbors[p_cellId * 27 + i];
            for(int j = 0; j < cellParticles[currentCellId]; ++j){
              int currentCellOffset = cellOffsets[currentCellId];
              if(currentCellOffset + j != idx){

                p_dist.x = p_pos.x - pos[currentCellOffset + j].x;
                p_dist.y = p_pos.y - pos[currentCellOffset + j].y;
                p_dist.z = p_pos.z - pos[currentCellOffset + j].z;
                distSqr = 0.0;
                distSqr += p_dist.x * p_dist.x;
                distSqr += p_dist.y * p_dist.y;
                distSqr += p_dist.z * p_dist.z;

                if(distSqr <= (cutoff * cutoff)){
                  if(calcMode == 0){
                    //gpuPots[0]._computeForceRaw(p_force, p_dist, distSqr);
                    double frac2 = 1.0 / distSqr;
                    double frac6 = frac2 * frac2 * frac2;
                    //double ffactor = frac6 * (gpuPots[0].ff1 * frac6 - gpuPots[0].ff2) * frac2;
                    double ffactor = frac6 * (48 * frac6 - 24) * frac2;
                    test_force.x += p_dist.x * ffactor;
                    test_force.y += p_dist.y * ffactor;
                    test_force.z += p_dist.z * ffactor;
                    //test_force.x += p_force.x;
                    //test_force.y += p_force.y;
                    //test_force.z += p_force.z;
  
                    //p_force.x = 0;
                    //p_force.y = 0;
                    //p_force.z = 0;
                  }
                  if(calcMode == 1){
                    //double frac2 = sigma*sigma / distSqr;
                    double frac2 = 1.0 * 1.0 / distSqr;
                    double frac6 = frac2 * frac2 * frac2;
                    //double energy = 4.0 * epsilon * (frac6 * frac6 - frac6);
                    double energy = 4.0 * 1.0 * (frac6 * frac6 - frac6);
                    p_energy += energy;
                    //p_energy += potential._computeEnergySqrRaw(distSqr);
                  }

                }
              }
            }
          }
//*/
/*
        }
      for(int i = 0; i < nPart; i++){
        if(i != idx){
          distSqr = 0.0;
          p_dist.x = p_pos.x - pos[i].x;
          p_dist.y = p_pos.y - pos[i].y;
          p_dist.z = p_pos.z - pos[i].z;
          distSqr += p_dist.x * p_dist.x;
          distSqr += p_dist.y * p_dist.y;
          distSqr += p_dist.z * p_dist.z;
          
          if(distSqr <= (cutoff * cutoff)){
            if(calcMode == 0){
              if(p_real == 1){
                gpuPots[0]._computeForceRaw(p_force, p_dist, distSqr);
                p_numForceCalc++;
                test_force.x += p_force.x;
                test_force.y += p_force.y;
                test_force.z += p_force.z;

                p_force.x = 0;
                p_force.y = 0;
                p_force.z = 0;
              }
            }
          }
        }
      }
*/
        }
        __syncthreads();
        if(calcMode == 0){
          force[idx].x = p_real * test_force.x;
          force[idx].y = p_real * test_force.y;
          force[idx].z = p_real * test_force.z;
        }
  
        if(calcMode == 1){
          energy[idx] = p_energy;
        }
      }
    }


  double LJGPUdriver(StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots, int mode){
    int numThreads = 256;
    int numBlocks = (gpuStorage->numberLocalParticles) / numThreads + 1;
    double *h_energy; 
    double *d_energy;
    double totalEnergy = 0;

    h_energy = new double[gpuStorage->numberLocalParticles];
    cudaMalloc(&d_energy, sizeof(double) * gpuStorage->numberLocalParticles);
    //printf("numLocalCells: %d, Particles: %d, numBlocks: %d, numThreads: %d\n", gpuStorage->numberLocalCells, gpuStorage->numberLocalParticles, numBlocks, numBlocks*numThreads);
    testKernel<<<numBlocks, numThreads>>>(
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
                            mode
                          );
      cudaDeviceSynchronize();  CUERR

      //printf("---\n");
      if(mode == 1) {
        cudaMemcpy(h_energy, d_energy, sizeof(double) * gpuStorage->numberLocalParticles, cudaMemcpyDeviceToHost);
        for (int i = 0; i < gpuStorage->numberLocalParticles; i++){
          totalEnergy += h_energy[i];
        }
      }
      return totalEnergy / 2;
    }
  }
}

