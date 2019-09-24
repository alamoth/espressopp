/*
  Copyright (C) 2012,2013,2014,2015,2016,2017,2018
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
#ifndef _INTERACTION_CELLLISTALLPARTICLESINTERACTIONTEMPLATEGPU_HPP
#define _INTERACTION_CELLLISTALLPARTICLESINTERACTIONTEMPLATEGPU_HPP

#include "types.hpp"
#include "Tensor.hpp"
#include "Interaction.hpp"
#include "storage/Storage.hpp"
#include "iterator/CellListIterator.hpp"
#include "esutil/Array2D.hpp"
//#include "LennardJonesGPU.cuh"

#define CUERR { \
    cudaError_t cudaerr; \
    if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
        printf("CUDA ERROR: \"%s\" in File %s at LINE %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__); \
    } \
}

namespace espressopp {
  namespace interaction {

    template < typename _Potential, typename _dPotential >
    class CellListAllParticlesInteractionTemplateGPU: public Interaction {
    protected:
      typedef _Potential Potential;
      typedef _dPotential dPotential;
    public:
      CellListAllParticlesInteractionTemplateGPU
      (shared_ptr < storage::Storage > _storage)
        : storage(_storage) {
          potentialArray    = esutil::Array2D<Potential, esutil::enlarge>(0, 0, Potential());
          ntypes = 0;
        }
      
      void
      setPotential(int type1, int type2, Potential &potential) {
        // typeX+1 because i<ntypes
        ntypes = std::max(ntypes, std::max(type1+1, type2+1));
        potentialArray.at(type1, type2) = potential;
        LOG4ESPP_INFO(_Potential::theLogger, "added potential for type1=" << type1 << " type2=" << type2);
        if (type1 != type2) { // add potential in the other direction
           potentialArray.at(type2, type1) = potential;
           LOG4ESPP_INFO(_Potential::theLogger, "automatically added the same potential for type1=" << type2 << " type2=" << type1);
        }

        potential.copyToGPUImplt(&h_potential);
        
        if(d_potentials != NULL){
           cudaFree(d_potentials); CUERR
        }
        
        int offset = potentialArray.size_x() * type2 + type1;
        printf("Offset: %d, size array: %d x %d, size Allocated space: %d\n", offset, potentialArray.size_x(), potentialArray.size_y()
         ,sizeof(dPotential) * potentialArray.size_x() * potentialArray.size_y());
        cudaMalloc(&d_potentials, sizeof(dPotential) * potentialArray.size_x() * potentialArray.size_y()); CUERR
        
        cudaMemcpy(&d_potentials[offset], &h_potential, sizeof(dPotential), cudaMemcpyHostToDevice); CUERR

        //cudaMalloc(&d_potentials, sizeof(Potential)); CUERR
        //cudaMemcpy(d_potentials, &potential, sizeof(Potential), cudaMemcpyHostToDevice); CUERR
        //gpuTest<Potential>(d_potentials);
        //gpuClass.testF();
        //gpuClass.gpuTest<Potential>(d_potentials);
        //gpu_addPotential(potential, type1, type2);
      }
      // this is used in the innermost force-loop
      Potential &getPotential(int type1, int type2) {
        return potentialArray.at(type1, type2);
      }

      // this is mainly used to access the potential from Python (e.g. to change parameters of the potential)
      shared_ptr<Potential> getPotentialPtr(int type1, int type2) {
    	return  make_shared<Potential>(potentialArray.at(type1, type2));
      }

      void testFunction(){
        //gpuClass.gpu_testGPU();
      }


      //virtual void testFF();
      virtual void addForces();
      virtual real computeEnergy();
      virtual real computeEnergyDeriv();
      virtual real computeEnergyAA();
      virtual real computeEnergyCG();
      virtual real computeEnergyAA(int atomtype);
      virtual real computeEnergyCG(int atomtype);
      virtual void computeVirialX(std::vector<real> &p_xx_total, int bins);
      virtual real computeVirial();
      virtual void computeVirialTensor(Tensor& wij);
      virtual void computeVirialTensor(Tensor& w, real z);
      virtual void computeVirialTensor(Tensor *w, int n);
      virtual real getMaxCutoff() { return 0.0; }
      virtual int bondType() { return Nonbonded; }

    protected:
      int ntypes;
      shared_ptr< storage::Storage > storage;
      shared_ptr< Potential > potential;
      esutil::Array2D<Potential, esutil::enlarge> potentialArray;
      dPotential h_potential;
      dPotential* d_potentials = 0;
      //gpu_CellListAllParticlesInteractionTemplateGPU<Potential> *gpuClass;
      //dPotential *d_potentials;


    };

    //////////////////////////////////////////////////
    // INLINE IMPLEMENTATION
    //////////////////////////////////////////////////
    template < typename _Potential, typename _dPotential > inline void
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    addForces() {
      LOG4ESPP_INFO(theLogger, "add forces computed for all particles in the cell lists");
      //printf("Size of type: %d, sizeof arrayiteself %d\n", sizeof(esutil::Array2D<Potential, esutil::enlarge>), sizeof(potentialArray));
      //printf("Size pot aray: %d, %d\n", potentialArray.size_n(), potentialArray.size_m());
      // TODO: one kernel call with all potentials
      //potential->testFF(&getPotential(1,1));
      potential->_computeForce(storage->getGPUstorage(), d_potentials);
    }

    template < typename _Potential, typename _dPotential >
    inline real
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeEnergy() {
      LOG4ESPP_INFO(theLogger, "compute energy for all particles in cell list");

      // for the long range interaction the energy is already reduced in _computeEnergy
      return potential->_computeEnergy(storage->getRealCells());
    }

    template < typename _Potential, typename _dPotential > inline real
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeEnergyDeriv() {
      std::cout << "Warning! At the moment computeEnergyDeriv() in CellListAllParticlesInteractionTemplateGPU does not work." << std::endl;
      return 0.0;
    }

    template < typename _Potential, typename _dPotential > inline real
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeEnergyAA() {
      std::cout << "Warning! At the moment computeEnergyAA() in CellListAllParticlesInteractionTemplateGPU does not work." << std::endl;
      return 0.0;
    }

    template < typename _Potential, typename _dPotential > inline real
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeEnergyAA(int atomtype) {
      std::cout << "Warning! At the moment computeEnergyAA(int atomtype) in CellListAllParticlesInteractionTemplateGPU does not work." << std::endl;
      return 0.0;
    }

    template < typename _Potential, typename _dPotential > inline real
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeEnergyCG() {
      std::cout << "Warning! At the moment computeEnergyCG() in CellListAllParticlesInteractionTemplateGPU does not work." << std::endl;
      return 0.0;
    }

    template < typename _Potential, typename _dPotential > inline real
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeEnergyCG(int atomtype) {
      std::cout << "Warning! At the moment computeEnergyCG(int atomtype) in CellListAllParticlesInteractionTemplateGPU does not work." << std::endl;
      return 0.0;
    }

    template < typename _Potential, typename _dPotential >
    inline void
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeVirialX(std::vector<real> &p_xx_total, int bins) {
        std::cout << "Warning! At the moment computeVirialX in CellListAllParticlesInteractionTemplateGPU does not work." << std::endl << "Therefore, the corresponding interactions won't be included in calculation." << std::endl;
    }

    template < typename _Potential, typename _dPotential > inline real
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeVirial() {
      LOG4ESPP_INFO(theLogger, "computed virial for all particles in the cell lists");

      // for the long range interaction the virial is already reduced in _computeVirial
      return potential -> _computeVirial(storage->getRealCells());
    }

    template < typename _Potential, typename _dPotential > inline void
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeVirialTensor(Tensor& wij) {
      LOG4ESPP_INFO(theLogger, "computed virial tensor for all particles in the cell lists");

      // for the long range interaction the virialTensor is already reduced in _computeVirialTensor
      //wij += potential -> _computeVirialTensor(storage->getRealCells());
    }


    template < typename _Potential, typename _dPotential > inline void
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeVirialTensor(Tensor& wij, real z) {
      std::cout<<"Warning! Calculating virial layerwise is not supported for "
              "long range interactions."<<std::endl;
    }

    template < typename _Potential, typename _dPotential > inline void
    CellListAllParticlesInteractionTemplateGPU <_Potential, _dPotential >::
    computeVirialTensor(Tensor *wij, int n) {
      std::cout<<"Warning! Calculating virial layerwise is not supported for "
              "long range interactions."<<std::endl;
    }

  }
}


#endif
