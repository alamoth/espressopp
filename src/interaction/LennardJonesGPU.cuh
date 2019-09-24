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


#include <cuda_runtime.h>

using namespace std;


namespace espressopp {
  namespace interaction {

    class d_LennardJonesGPU {
    public:

      double epsilon;
      double sigma;
      double ff1, ff2;
      double ef1, ef2;

      
    
      d_LennardJonesGPU()
	      : epsilon(0.0), sigma(0.0) {
          preset();
      }

      d_LennardJonesGPU(double _epsilon, double _sigma) {
          preset();
      }
      
      ~d_LennardJonesGPU(){};

      void preset() {
        double sig2 = sigma * sigma;
        double sig6 = sig2 * sig2 * sig2;
        ff1 = 48.0 * epsilon * sig6 * sig6;
        ff2 = 24.0 * epsilon * sig6;
        ef1 =  4.0 * epsilon * sig6 * sig6;
        ef2 =  4.0 * epsilon * sig6;
      }

      // Setter and getter
      void setEpsilon(double _epsilon) {
        epsilon = _epsilon;
        preset();
      }
      
      double getEpsilon() const { return epsilon; }

      __host__ __device__ void setSigma(double _sigma) { 
        sigma = _sigma; 
        preset();
      }
      double getSigma() const { return sigma; }
      
      bool _computeForceRaw(double3& force, const double3& dist, double distSqr){

        double frac2 = 1.0 / distSqr;
        double frac6 = frac2 * frac2 * frac2;
        double ffactor = frac6 * (ff1 * frac6 - ff2) * frac2;
        //force = dist * ffactor;

        return true;
      }

      double _computeEnergySqrRaw(double distSqr){
        
        
        return 0.0;
      }

    };

    void LJGPUdriver( int nPart,
                      int nCells,
                      double3* pos, 
                      double3* force,
                      double* mass,
                      double* drift,
                      int* type,
                      int* cellOff,
                      int* numCellN,
                      d_LennardJonesGPU* gpuPots);
  }
}

#endif
