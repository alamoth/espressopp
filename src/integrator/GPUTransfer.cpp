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
#include <iterator>
#include <vector>

namespace espressopp {

  using namespace iterator;

  namespace integrator {

    LOG4ESPP_LOGGER(GPUTransfer::theLogger, "GPUTransfer");

    GPUTransfer::GPUTransfer(shared_ptr<System> system)
    : Extension(system)
    {
      printf("Constructor GPUTransfer\n");
      // Initialize GPU
      System& _system = getSystemRef();
      StorageGPU* GPUStorage = _system.storage->getGPUstorage();
    }

    void GPUTransfer::disconnect(){
      _onParticlesChanged.disconnect();
      _aftInitF.disconnect();
      _aftCalcF.disconnect();
      _runInit.disconnect();
    }

    void GPUTransfer::connect(){
      System& system = getSystemRef();
      _onParticlesChanged   =   system.storage->onParticlesChanged.connect( boost::bind(&GPUTransfer::onDecompose, this));
      _aftInitF             =   integrator->aftInitF.connect( boost::bind(&GPUTransfer::fillParticleVars, this));
  	  _aftCalcF             =   integrator->aftCalcF.connect( boost::bind(&GPUTransfer::getParticleForces, this));
      _runInit              =   integrator->runInit.connect ( boost::bind(&GPUTransfer::onRunInit, this));
    }


    // Cell Data
    //
    void GPUTransfer::onRunInit(){
      GPUTransfer::resizeCellData();
      GPUTransfer::fillCellData();
    }

    void GPUTransfer::resizeCellData(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      CellList localCells = system.storage->getLocalCells();

      GPUStorage->numberCells = localCells.size();
      GPUStorage->resizeCellData();
    }

    void GPUTransfer::fillCellData(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      CellList localCells = system.storage->getLocalCells();
      int counterParticles = 0;
      for(unsigned int i = 0; i < localCells.size(); ++i) {
        GPUStorage->h_cellOffsets[i] = counterParticles;
        GPUStorage->h_numberCellNeighbors[i] = localCells[i]->neighborCells.size() == 0 ? 0 : 1;
        counterParticles += localCells[i]->particles.size();
      }
      GPUStorage->h2dCellData();
    }
    // Cell Data end

    // On decompose, resize Particle Arrays and fill with statics
    //
    void GPUTransfer::onDecompose(){
      GPUTransfer::resizeParticleData();
      GPUTransfer::fillParticleStatics();
    }

    void GPUTransfer::resizeParticleData(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      GPUStorage->numberParticles = system.storage->getNLocalParticles();
      GPUStorage->resizeParticleData();
    }

    void GPUTransfer::fillParticleStatics(){
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();   

      unsigned int counterParticles = 0;
      for(unsigned int i = 0; i < localCells.size(); ++i) {
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          Particle &p = localCells[i]->particles[j];
          GPUStorage->h_type[counterParticles] = p.getType();
          GPUStorage->h_mass[counterParticles] = p.getMass();
          GPUStorage->h_drift[counterParticles] = p.getDrift();
          counterParticles++;
        }
      }
      GPUStorage->h2dParticleStatics();
    }
    // On decompose end

    void GPUTransfer::fillParticleVars(){
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      unsigned int counterParticles = 0;

      for(unsigned int i = 0; i < localCells.size(); ++i) {
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          Real3D pos = localCells[i]->particles[j].getPos();
          GPUStorage->h_pos[counterParticles] = make_double3(pos.at(0), pos.at(1), pos.at(2)); 
          //GPUStorage->h_px[counterParticles] = pos.at(0);
          //GPUStorage->h_py[counterParticles] = pos.at(1);
          //GPUStorage->h_pz[counterParticles] = pos.at(2);
          counterParticles++;
        }
      }
      GPUStorage->h2dParticleVars();
    }

    void GPUTransfer::getParticleForces(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      CellList localCells = system.storage->getLocalCells();

      GPUStorage->d2hParticleForces();
      
      unsigned int counterParticles = 0;
      

      for(unsigned int i = 0; i < localCells.size(); ++i) {
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          double3 force3 = GPUStorage->h_force[counterParticles];
          Real3D force3D(force3.x, force3.y, force3.z);
          Particle &p = localCells[i]->particles[j];
          p.force() += force3D;
          //printf("Force copies back: %f, %f, %f\n", force3.x, force3.y, force3.z);
        }
      }
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
