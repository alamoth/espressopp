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
                double3* pos,
                double3* force,
                double* mass,
                double* drift,
                int* type,
                int* cellOff,
                int* numCellN, 
                bool* real,
                double* energy,
                d_LennardJonesGPU* gpuPots,
                int mode){
      int idx = blockIdx.x*blockDim.x + threadIdx.x;

      __shared__ double cutoff;
      __shared__ int calcMode;
      __shared__ d_LennardJonesGPU potential;

      if(threadIdx.x == 0){
        //cutoff = gpuPots[0].getCutoff();
        cutoff = 2.9f;
        calcMode = mode;
        potential = gpuPots[0];
        //printf("Cutoff: %f, mode: %d, potential.sigma: %f\n", cutoff, calcMode, potential.getSigma());
      }
      __syncthreads();

      double3 p_pos = pos[idx];
      double p_mass = mass[idx];
      double p_drift = drift[idx];
      int p_id = id[idx];
      int p_type = type[idx];
      int p_real = real[idx] ? 1 : 0;
      double3 p_force = make_double3(0.0,0.0,0.0);
      double3 test_force = make_double3(0.0,0.0,0.0);
      double3 p_dist;
      double distSqr;
      double p_energy = 0;

      if(idx < nPart){
        //printf("Cutoff: %f\n", gpuPots[0].cutoff);
        //printf("Particle %d, real: %s, pos: x: %f, y: %f, z: %f\n", idx, real[idx] ? "true" : "false", pos[idx].x, pos[idx].y, pos[idx].z);
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

                  test_force.x += p_force.x;
                  test_force.y += p_force.y;
                  test_force.z += p_force.z;

                  p_force.x = 0;
                  p_force.y = 0;
                  p_force.z = 0;
                }
              }
                  //printf("-\n");
                  //printf("real? %d\n", p_real);
                  //printf("thread idx: %d\np1 x: %f, y: %f, z: %f;  p2 x: %f, y: %f, z: %f, p1 real: %d, p2 real: %d\np1: id: %d, p2 id: %d, force x: %f, x: %f, z: %f\ndistSqr: %f, p_dist: x: %f, y: %f, z:%f\n\n" ,  idx, p_pos.x, p_pos.y, p_pos.z, pos[i].x, pos[i].y, pos[i].z, p_real, real[i], p_id, id[i], p_force.x, p_force.y, p_force.z, distSqr, p_dist.x, p_dist.y, p_dist.z);

            }
          }

          if(calcMode == 1){
            p_energy += potential._computeEnergySqrRaw(distSqr);
          }

        }
      
        // printf("Force x: %f, y: %f, z: %f\n", p_force.x, p_force.y, p_force.z);
        if(calcMode == 0){
          force[idx].x = p_real * test_force.x;
          force[idx].y = p_real * test_force.y;
          force[idx].z = p_real * test_force.z;
        }

        if(calcMode == 1){
          energy[idx] = p_energy;
          //printf("Energy p %d: %f\n", idx, energy[idx]);
        }
      }
      __syncthreads();

    }

/*
    void LJGPUdriver( int nPart,
                      int nCells,
                      double3* pos, 
                      double3* force,
                      double* mass,
                      double* drift,
                      int* type,
                      int* cellOff,
                      int* numCellN,
                      d_LennardJonesGPU* gpuPots){
*/
  double LJGPUdriver(StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots, int mode){
    //printf("cutof: %f\n", gpuPots[0].sigma);
    int numThreads = 128;
    int numBlocks = (gpuStorage->numberLocalParticles) / numThreads + 1;
    double *h_energy; 
    double *d_energy;
    double totalEnergy = 0;

    //h_energy = new double[gpuStorage->numberLocalParticles];
    //cudaMalloc(&d_energy, sizeof(double) * gpuStorage->numberLocalParticles);
    //printf("Particle: %d, numBlocks: %d\n", gpuStorage->numberLocalParticles, numBlocks);
    testKernel<<<numBlocks, numThreads>>>(
                            gpuStorage->numberLocalParticles, 
                            gpuStorage->numberLocalCells, 
                            gpuStorage->d_id,
                            gpuStorage->d_pos,
                            gpuStorage->d_force,
                            gpuStorage->d_mass,
                            gpuStorage->d_drift,
                            gpuStorage->d_type,
                            gpuStorage->d_cellOffsets,
                            gpuStorage->d_numberCellNeighbors,
                            gpuStorage->d_real,
                            d_energy,
                            gpuPots,
                            mode
                          );
      cudaDeviceSynchronize();  CUERR

      //printf("---\n");
      if(mode == 1) {
        printf("if you see this you failed");
        cudaMemcpy(h_energy, d_energy, sizeof(double) * gpuStorage->numberLocalParticles, cudaMemcpyDeviceToHost);
        for (int i = 0; i < gpuStorage->numberLocalParticles; i++){
          totalEnergy += h_energy[i];
        }
      }
      return totalEnergy / 2;
    }
  }
}

