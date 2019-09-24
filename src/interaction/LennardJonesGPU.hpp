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
#ifndef _INTERACTION_LENNARDJONESGPU_HPP
#define _INTERACTION_LENNARDJONESGPU_HPP

#include <cmath>

#include "storage/Storage.hpp"

#include "Potential.hpp"
#include <cuda_runtime.h>
#include "LennardJonesGPU.cuh"


using namespace std;


namespace espressopp {
  namespace interaction {

    
    class LennardJonesGPU : public PotentialTemplate< LennardJonesGPU > {
    protected:

      real epsilon;
      real sigma;
      real ff1, ff2;
      real ef1, ef2;

    public:
      static void registerPython();

      LennardJonesGPU()
	      : epsilon(0.0), sigma(0.0) {
          setShift(0.0);
          setCutoff(infinity);
          preset();
      }

      LennardJonesGPU(real _epsilon, real _sigma, 
		    real _cutoff, real _shift) 
	        : epsilon(_epsilon), sigma(_sigma) {
          setShift(_shift);
          setCutoff(_cutoff);
          preset();
      }

      LennardJonesGPU(real _epsilon, real _sigma, 
		    real _cutoff)
	        : epsilon(_epsilon), sigma(_sigma) {	
          autoShift = false;
          setCutoff(_cutoff);
          preset();
          setAutoShift(); 
      }
      
      virtual ~LennardJonesGPU(){};

      void copyToGPUImplt(d_LennardJonesGPU* h_potential){
        h_potential->setEpsilon(epsilon);
        h_potential->setSigma(sigma);
      }

      void preset() {
        real sig2 = sigma * sigma;
        real sig6 = sig2 * sig2 * sig2;
        ff1 = 48.0 * epsilon * sig6 * sig6;
        ff2 = 24.0 * epsilon * sig6;
        ef1 =  4.0 * epsilon * sig6 * sig6;
        ef2 =  4.0 * epsilon * sig6;
      }

      // Setter and getter
      void setEpsilon(real _epsilon) {
        epsilon = _epsilon;
        LOG4ESPP_INFO(theLogger, "epsilon=" << epsilon);
        updateAutoShift();
        preset();
      }
      
      real getEpsilon() const { return epsilon; }

      void setSigma(real _sigma) { 
        sigma = _sigma; 
        LOG4ESPP_INFO(theLogger, "sigma=" << sigma);
        updateAutoShift();
        preset();
      }
      real getSigma() const { return sigma; }
      
      bool _computeForce(StorageGPU *gpuStorage, d_LennardJonesGPU *gpuPots){
        // TOdo: GPU
        // printf("SIzeofLennardJones: %d\n", sizeof(potentialArray));
        //gpu_computeForce(d_potential);
        //double3 testPos = gpuStorage->h_pos[0];
        //printf("h_pos[0].x: %f, y: %f, z: %f\n", testPos.x, testPos.y, testPos.z);
        LJGPUdriver(  gpuStorage->numberParticles, 
                      gpuStorage->numberCells, 
                      gpuStorage->d_pos,
                      gpuStorage->d_force,
                      gpuStorage->d_mass,
                      gpuStorage->d_drift,
                      gpuStorage->d_type,
                      gpuStorage->d_cellOffsets,
                      gpuStorage->d_numberCellNeighbors,
                      gpuPots
                      );
                      
        return true;
      }

      real _computeEnergy(CellList realcells){
        
        
        return 0.0;
      }


      real _computeVirial(CellList realcells){
        
          return 0.0;
      }
      

      void _computeVirialTensor(CellList realcells){

      }
      
      real _computeEnergySqrRaw(real distSqr) const {
        
        return 0.0;
      }
      bool _computeForceRaw(Real3D& force, const Real3D& dist, real distSqr) const {
        
        return false;
      }
      
    protected:
      
    };
  }
}

#endif
