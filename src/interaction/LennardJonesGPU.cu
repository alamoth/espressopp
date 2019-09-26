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

using namespace std;



namespace espressopp {
  namespace interaction {

    __global__ void 
    testKernel( int nPart,
                int nCells,
                double3* pos,
                double3* force,
                double* mass,
                double* drift,
                int* type,
                int* cellOff,
                int* numCellN, 
                d_LennardJonesGPU* gpuPots){
      int idx = blockIdx.x*blockDim.x + threadIdx.x;
      
      if(idx < 1){
        printf("Cutoff: %f\n", gpuPots[0].cutoff);
        printf("#Cells: %d, cellOff[%d]: %d, numCell: %d\n", nCells, idx, cellOff[idx], numCellN[idx]);
      }

      if(idx < nPart){
        //printf("PID: %d, type: %d, mass: %f, drift: %f\n", idx, type[idx], mass[idx], drift[idx]);
        //printf("d_pos[0].x: %f, y: %f, z: %f\n", pos[0].x, pos[0].y, pos[0].z);
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
  void LJGPUdriver(StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots){
    //printf("cutof: %f\n", gpuPots[0].sigma);
    testKernel<<<1,100>>>(  gpuStorage->numberParticles, 
                            gpuStorage->numberCells, 
                            gpuStorage->d_pos,
                            gpuStorage->d_force,
                            gpuStorage->d_mass,
                            gpuStorage->d_drift,
                            gpuStorage->d_type,
                            gpuStorage->d_cellOffsets,
                            gpuStorage->d_numberCellNeighbors,
                            gpuPots);

    }
  }
}

