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
                bool* ghost,
                double* energy,
                d_LennardJonesGPU* gpuPots,
                int mode){
      int idx = blockIdx.x*blockDim.x + threadIdx.x;

      __shared__ double cutoff;
      __shared__ int calcMode;

      if(idx == 0){
        cutoff = gpuPots[0].getCutoff();
        calcMode = mode;
      }
      __syncthreads();

      double3 p_pos = pos[idx];
      double p_mass = mass[idx];
      double p_drift = drift[idx];
      int p_id = id[idx];
      int p_type = type[idx];
      int p_ghost = ghost[idx] ? 0 : 1;
      double3 p_force = make_double3(0.0,0.0,0.0);
      double3 p_dist;
      double distSqr;
      double p_energy = 0;

      if(idx < nPart){
        //printf("Cutoff: %f\n", gpuPots[0].cutoff);
        //printf("Particle %d, Ghost: %s, pos: x: %f, y: %f, z: %f\n", idx, ghost[idx] ? "true" : "false", pos[idx].x, pos[idx].y, pos[idx].z);
        for(int i = 0; i < nPart; i++){
          if(i != idx){
            distSqr = 0.0;
            p_dist.x = p_pos.x - pos[i].x;
            p_dist.y = p_pos.y - pos[i].y;
            p_dist.z = p_pos.z - pos[i].z;
            distSqr += p_dist.x * p_dist.x;
            distSqr += p_dist.y * p_dist.y;
            distSqr += p_dist.z * p_dist.z;
            //printf("distSqr: %f, p_dist: x: %f, y: %f, z:%f\n", distSqr, p_dist.x, p_dist.y, p_dist.z);
            if(distSqr < (cutoff * cutoff)){
              if(calcMode == 0){
                gpuPots[0]._computeForceRaw(p_force, p_dist, distSqr);
                if(p_ghost == 1){
                  //printf("-\n");
                  //printf("ghost? %d\n", p_ghost);
                  printf("p1: id: %d, x: %f, y: %f, z: %f;  p2: id: %d, x: %f, y: %f, z: %f\n", p_id, p_pos.x,p_pos.y,p_pos.z, id[i], pos[i].x, pos[i].y, pos[i].z);
                  //printf("id1: %d, id2: %d, force x: %f, y: %f, z: %f\n", p_id, id[i], p_force.x, p_force.y, p_force.z);
                }
              }
              if(calcMode == 1){
                p_energy += gpuPots[0]._computeEnergySqrRaw(distSqr);
              }
            }
          }
        }
        // printf("Force x: %f, y: %f, z: %f\n", p_force.x, p_force.y, p_force.z);
        if(calcMode == 0){
          force[idx].x = p_ghost * p_force.x;
          force[idx].y = p_ghost * p_force.y;
          force[idx].z = p_ghost * p_force.z;
        }

        if(calcMode == 1){
          energy[idx] = p_energy;
          //printf("Energy p %d: %f\n", idx, energy[idx]);
        }
      }
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
    int numBlocks = gpuStorage->numberLocalParticles / 512 + 1;
    double *h_energy; 
    double *d_energy;
    double totalEnergy = 0;

    h_energy = new double[gpuStorage->numberLocalParticles];
    cudaMalloc(&d_energy, sizeof(double) * gpuStorage->numberLocalParticles);

    testKernel<<<numBlocks, 512>>>(
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
                            gpuStorage->d_ghost,
                            d_energy,
                            gpuPots,
                            mode
                          );

      printf("---\n");
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

