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

#ifndef LennardJonesGPU_CUH
#define LennardJonesGPU_CUH
#include <cmath>
#include <stdio.h>
#include "stdio.h"
// #include "esutil/CudaHelper.cuh"


#include <cuda_runtime.h>
#include "storage/StorageGPU.hpp"
using namespace std;

namespace espressopp {
  namespace interaction {

    class d_LennardJonesGPU {
    public:

      realG epsilon;
      realG sigma;
      realG ff1, ff2;
      realG ef1, ef2;
      realG cutoff;

      
    //public:
      d_LennardJonesGPU()
	      : epsilon(0.0), sigma(0.0) {
          preset();
      }

      d_LennardJonesGPU(realG _epsilon, realG _sigma) {
          preset();
      }
      
      ~d_LennardJonesGPU(){};

      __device__ __host__ 
      void preset() {
        realG sig2 = sigma * sigma;
        realG sig6 = sig2 * sig2 * sig2;
        ff1 = 48.0 * epsilon * sig6 * sig6;
        ff2 = 24.0 * epsilon * sig6;
        ef1 =  4.0 * epsilon * sig6 * sig6;
        ef2 =  4.0 * epsilon * sig6;
      }

      // Setter and getter
      void setEpsilon(realG _epsilon) {
        epsilon = _epsilon;
        preset();
      }

      __device__ __host__ 
      realG getEpsilon() const { return epsilon; }

      void setSigma(realG _sigma) { 
        sigma = _sigma; 
        preset();
      }

      __device__ __host__ 
      realG getSigma() const { return sigma; }
      
      void setCutoff(realG _cutoff) {
        cutoff = _cutoff;
        preset();
      }

      __device__ __host__
      realG getCutoff() const { return cutoff; }
    };
  /*
    void LJGPUdriver( int nPart,
                      int nCells,
                      realG3* pos, 
                      realG3* force,
                      realG* mass,
                      realG* drift,
                      int* type,
                      int* cellOff,
                      int* numCellN,
                      d_LennardJonesGPU* gpuPots);
                      */
    realG LJGPUdriver (StorageGPU* gpuStorage, d_LennardJonesGPU* gpuPots, int mode);
  }
}

#endif
