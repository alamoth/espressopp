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
#include "MortonHelper.h"
#include "GPUTransfer.cuh"
#include <math.h>

namespace espressopp {

  using namespace iterator;

  namespace integrator {


    LOG4ESPP_LOGGER(GPUTransfer::theLogger, "GPUTransfer");

    GPUTransfer::GPUTransfer(shared_ptr<System> system)
    : Extension(system)
    {
      // int rank;
      // int size;
      // MPI_Comm_rank( MPI_COMM_WORLD, &rank );
      // MPI_Comm_rank( MPI_COMM_WORLD, &size );
      // cudaError_t err;
      // int dev_cnt = 0;
      // err = cudaGetDeviceCount( &dev_cnt );
      // assert( err == cudaSuccess || err == cudaErrorNoDevice );
      // printf( "rank %d, mpi size: %d, cnt %d\n", rank, size, dev_cnt );
      
      // cudaDeviceProp prop;
      // for (int dev = 0; dev < dev_cnt; ++dev) {
      //     err = cudaGetDeviceProperties( &prop, dev );
      //     assert( err == cudaSuccess );
      //     printf( "rank %d, dev %d, prop %s, pci %d, %d, %d\n",
      //             rank, dev,
      //             prop.name,
      //             prop.pciBusID,
      //             prop.pciDeviceID,
      //             prop.pciDomainID );
      //     // printf("unique id: %.*s\n", (int)sizeof(prop.uuid), prop.uuid);
      // }

    }

    void GPUTransfer::disconnect(){
      _onParticlesChanged.disconnect();
      _aftInitF.disconnect();
      _aftCalcF.disconnect();
      _runInit.disconnect();
    }

    void GPUTransfer::connect(){
      System& system = getSystemRef();
      _onParticlesChanged   =   system.storage->gpuParticlesChanged.connect( boost::bind(&GPUTransfer::onDecompose, this));
      // _onParticlesChanged   =   integrator->gpuAftDec.connect( boost::bind(&GPUTransfer::onDecompose, this));
      _aftInitF             =   integrator->gpuBefF.connect( boost::bind(&GPUTransfer::fillParticleVars, this));
  	  _aftCalcF             =   integrator->gpuAftF.connect( boost::bind(&GPUTransfer::getParticleForces, this));
      _runInit              =   integrator->runInit.connect ( boost::bind(&GPUTransfer::onRunInit, this));
      _aftCellAdjust        =   system.storage->aftCellAdjust.connect ( boost::bind(&GPUTransfer::onRunInit, this));
    }


    // Cell Data
    //
    void GPUTransfer::onRunInit(){
      GPUTransfer::resizeCellData();
      GPUTransfer::fillCellData();
      GPUTransfer::onDecompose();
    }

    void GPUTransfer::resizeCellData(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      GPUStorage->numberLocalCells = system.storage->getLocalCells().size();
      GPUStorage->resizeCellData();
    }

    void GPUTransfer::fillCellData(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      CellList localCells = system.storage->getLocalCells();
      
      int counterParticles = 0;
      bool realCell;
      int3 mappedPos;
      for(unsigned int i = 0; i < localCells.size(); ++i) {
        realCell = localCells[i]->neighborCells.size() == 0 ? false : true;
        if(realCell){
          for(unsigned int j = 0; j < 13; ++j){
            GPUStorage->h_cellNeighbors[i * 27 + j] = localCells[i]->neighborCells[j].cell->id;
          }
          GPUStorage->h_cellNeighbors[i * 27 + 13] = localCells[i]->id;
          for(unsigned int j = 13; j < 26; ++j){
            GPUStorage->h_cellNeighbors[i * 27 + j + 1] = localCells[i]->neighborCells[j].cell->id;
          }
        }
        counterParticles += localCells[i]->particles.size(); // needed?
      }
      GPUStorage->h2dCellData();
    }
    // Cell Data end

    // On decompose, resize Particle Arrays and fill with statics
    //
    void GPUTransfer::onDecompose(){
      GPUTransfer::resizeParticleData();
      GPUTransfer::fillParticleStatics();
      GPUTransfer::fillParticleVars();
    }

    void GPUTransfer::resizeParticleData(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      GPUStorage->numberLocalParticles = system.storage->getNLocalParticles();
      GPUStorage->resizeParticleData();
    }

    void GPUTransfer::fillParticleStatics(){
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();   
      unsigned int counterParticles = 0;
      int max_n_nb = 0;
      bool realParticle;
      int n_cell_pt;
      for(unsigned int i = 0; i < localCells.size(); ++i) {
        realParticle = localCells[i]->neighborCells.size() == 0 ? false : true;
        n_cell_pt = localCells[i]->particles.size();
        GPUStorage->h_cellOffsets[i] = counterParticles;
        GPUStorage->h_particlesCell[i] = n_cell_pt;
        max_n_nb = n_cell_pt > max_n_nb ? n_cell_pt : max_n_nb;
        //printf("Offset[%d]: %d, cellParticles[%d]=%d\n",i,counterParticles,i, localCells[i]->particles.size());
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          Particle &p = localCells[i]->particles[j];
          GPUStorage->h_id[counterParticles] = p.getId();
          GPUStorage->h_cellId[counterParticles] = localCells[i]->id;
          GPUStorage->h_type[counterParticles] = p.getType();
          GPUStorage->h_mass[counterParticles] = p.getMass();
          GPUStorage->h_drift[counterParticles] = p.getDrift();
          GPUStorage->h_real[counterParticles] = realParticle;
          counterParticles++;
        }
      }
      GPUStorage->max_n_nb = max_n_nb;
      GPUStorage->h2dParticleStatics();
    }
    // On decompose end

    // Copy particle Pos in each time step
    void GPUTransfer::fillParticleVars(){
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      unsigned int counterParticles = 0;

      for(unsigned int i = 0; i < localCells.size(); ++i) {
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          Real3D pos = localCells[i]->particles[j].getPos();
          GPUStorage->h_pos[counterParticles] = make_realG3(pos.at(0), pos.at(1), pos.at(2), 0.0);
          counterParticles++;
        }
      }
      GPUStorage->h2dParticleVars();
    }

    // Copy forces back in each time step
    void GPUTransfer::getParticleForces(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      CellList localCells = system.storage->getLocalCells();

      GPUStorage->d2hParticleForces();
      
      unsigned int counterParticles = 0;

      for(unsigned int i = 0; i < localCells.size(); ++i) {
        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          realG3 force3 = GPUStorage->h_force[counterParticles];
          Real3D force3D(force3.x, force3.y, force3.z);
          Particle &p = localCells[i]->particles[j];
          p.force() += force3D;
          counterParticles++;
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
