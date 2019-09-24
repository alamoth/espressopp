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
/*
    __global__ void 
    testKernel(double3 *pos){
      printf("d_pos[0].x: %f, y: %f, z: %f\n", pos[0].x, pos[0].y, pos[0].z);
    }


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
      testKernel<<<1,1>>>(pos);
    }
  }
}

namespace espressopp {
  namespace interaction {

    void d_LennardJonesGPU::testFF(d_LennardJonesGPU* potential){
      printf("bna %f\n", potential->getSigma());
    }
    */
  }
}

