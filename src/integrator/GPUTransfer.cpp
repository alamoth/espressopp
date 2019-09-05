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

#include "python.hpp"
#include "GPUTransfer.hpp"

#include "types.hpp"
#include "System.hpp"
#include "storage/Storage.hpp"
#include "iterator/CellListIterator.hpp"

namespace espressopp {

  using namespace iterator;

  namespace integrator {

    LOG4ESPP_LOGGER(GPUTransfer::theLogger, "GPUTransfer");

    GPUTransfer::GPUTransfer(shared_ptr<System> system)
    : Extension(system)
    {
      // Initialize GPU
      System& _system = getSystemRef();
      StorageGPU* GPUStorage = _system.storage->getGPUstorage();
      
      GPUStorage->initNullPtr();
      GPUStorage->numberCells = _system.storage->getLocalCells().size();
      GPUStorage->allocateCellData();
    }

    void GPUTransfer::disconnect(){
      _onParticlesChanged.disconnect();
      _aftInitF.disconnect();
      _aftCalcF.disconnect();
    }

    void GPUTransfer::connect(){
      System& system = getSystemRef();

      _onParticlesChanged = system.storage->onParticlesChanged.connect( boost::bind(&GPUTransfer::ParticleStatics, this));
      _aftInitF = integrator->aftInitF.connect( boost::bind(&GPUTransfer::ParticleVars, this));
  	  _aftCalcF = integrator->aftCalcF.connect( boost::bind(&GPUTransfer::ParticleForces, this));
    }

    void GPUTransfer::ParticleVars(){
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      unsigned int counterParticles = 0;
      for(unsigned int i = 0; i < localCells.size(); ++i) {
        GPUStorage->h_cellOffsets[i] = counterParticles;
        GPUStorage->h_numberCellNeighbors[i] = localCells[i]->neighborCells.size() == 0 ? 0 : 1;
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          GPUStorage->h_px[counterParticles] = localCells[i]->particles[j].getPos().at(0);
          GPUStorage->h_py[counterParticles] = localCells[i]->particles[j].getPos().at(1);
          GPUStorage->h_pz[counterParticles] = localCells[i]->particles[j].getPos().at(2);
          counterParticles++;
        }
      }
      GPUStorage->h2dParticleVars();
      //printf("Size localCells: %ld\n", localCells.size());
    }

    void GPUTransfer::ParticleStatics(){
      //printf("ParticleStatics signal\n");
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      // Since there was a rebuild, allocate particle data new
      GPUStorage->numberParticles = system.storage->getNLocalParticles();
      GPUStorage->allocateParticleData(); // Could do resize and allocate only in constructor

      // Fill Particle static data
      unsigned int counterParticles = 0;
      for(unsigned int i = 0; i < localCells.size(); ++i) {
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          GPUStorage->h_type[counterParticles] = localCells[i]->particles[j].getType();
          GPUStorage->h_mass[counterParticles] = localCells[i]->particles[j].getMass();
          GPUStorage->h_drift[counterParticles] = localCells[i]->particles[j].getDrift();
          counterParticles++;
        }
      }
      GPUStorage->h2dParticleStatics();
    }

    void GPUTransfer::ParticleForces(){
      // printf("Signal aftCalcF\n");
      //GPUStorage->freeParticleVars();
    }

    /****************************************************
    ** REGISTRATION WITH PYTHON
    ****************************************************/

    void GPUTransfer::registerPython() {

      using namespace espressopp::python;

      class_<GPUTransfer, shared_ptr<GPUTransfer>, bases<Extension> >

        ("integrator_GPUTransfer", init< shared_ptr< System > >())
        .def("connect", &GPUTransfer::connect)
        .def("disconnect", &GPUTransfer::disconnect)
        ;
    }

  }
}
