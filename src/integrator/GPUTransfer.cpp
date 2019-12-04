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
// #include "MortonHelper.h"
#include "tools/libmorton/morton.h"
#include <math.h>
#include <inttypes.h>
#include <algorithm>
#include <chrono>

namespace espressopp {

  using namespace iterator;

  namespace integrator {



    LOG4ESPP_LOGGER(GPUTransfer::theLogger, "GPUTransfer");

    GPUTransfer::GPUTransfer(shared_ptr<System> system)
    : Extension(system)
    {
      timeTotal;
      timeFillParticleVars = 0;
      timeFillParticleStatics = 0;
      timeFillCellData = 0;
      timeGetParticleForces = 0;
      timeResizeParticleData = 0;
      timeResizeCellData = 0;

      time = 0;
      timeGResizeParticleData = 0;
      timeGResizeCellData = 0;
      timeGH2dParticleStatics = 0;
      timeGH2dParticleVars = 0;
      timeGH2dCellData = 0;
      timeGD2hParticleForces = 0;
      timeIntegrate.reset();

      cOnDecompose = 0;
      cOnGridInit = 0;
      cFillParticleVars = 0;
      cFillParticleStatics = 0;
      cFillCellData = 0;
      cGetParticleForces = 0;
      cResizeParticleData = 0;
      cResizeCellData = 0;

      int* d_init;
      
      time = timeIntegrate.getElapsedTime();
      cudaMalloc(&d_init, 1);
      timeGPUinit += time = timeIntegrate.getElapsedTime(); - time;

      int rank;
      int size;
      MPI_Comm_rank( MPI_COMM_WORLD, &rank );
      MPI_Comm_rank( MPI_COMM_WORLD, &size );
      cudaError_t err;
      int dev_cnt = 0;
      err = cudaGetDeviceCount( &dev_cnt );
      assert( err == cudaSuccess || err == cudaErrorNoDevice );
      // // printf( "rank %d, mpi size: %d, cnt %d\n", rank, size, dev_cnt );
      
      cudaDeviceProp prop;
      int device;
      cudaGetDevice(&device);
      err = cudaGetDeviceProperties( &prop, device );
      assert( err == cudaSuccess );
      printf( "rank %d, device %d, prop %s, pci %d, %d, %d\n",
              rank,
              device,
              prop.name,
              prop.pciBusID,
              prop.pciDeviceID,
              prop.pciDomainID );
      // for (int dev = 0; dev < dev_cnt; ++dev) {
      //     err = cudaGetDeviceProperties( &prop, dev );
      //     assert( err == cudaSuccess );
      //     printf( "rank %d, dev %d, prop %s, pci %d, %d, %d\n",
      //             rank, dev,
      //             prop.name,
      //             prop.pciBusID,
      //             prop.pciDeviceID,
      //             prop.pciDomainID );
          // printf("unique id: %.*s\n", (int)sizeof(prop.uuid), prop.uuid);
      // }
      // cudaSetDevice(6);

    }

    void GPUTransfer::disconnect(){
      _onParticlesChanged.disconnect();
      _aftInitF.disconnect();
      _aftCalcF.disconnect();
      _runInit.disconnect();
      GPUTransfer::printTimers();
    }

    void GPUTransfer::connect(){
      System& system = getSystemRef();
      _onParticlesChanged   =   system.storage->gpuParticlesChanged.connect( boost::bind(&GPUTransfer::onDecompose, this));
      // _onParticlesChanged   =   integrator->gpuAftDec.connect( boost::bind(&GPUTransfer::onDecompose, this));
      _aftInitF             =   integrator->gpuBefF.connect( boost::bind(&GPUTransfer::fillParticleVars, this));
  	  _aftCalcF             =   integrator->gpuAftF.connect( boost::bind(&GPUTransfer::getParticleForces, this));
      //_runInit              =   integrator->runInit.connect ( boost::bind(&GPUTransfer::onRunInit, this)); 
      _aftCellAdjust        =   system.storage->aftCellAdjust.connect ( boost::bind(&GPUTransfer::onGridInit, this));
      // _aftGridInit        =   system.storage->aftCellAdjust.connect ( boost::bind(&GPUTransfer::onRunInit, this));
      GPUTransfer::onGridInit();
    }


    // Cell Data
    //
    void GPUTransfer::onGridInit(){
      // printf("onGridInit\n");
      GPUTransfer::resizeCellData();
      GPUTransfer::fillCellData();

      cOnGridInit++;
    }
    // void GPUTransfer::onRunInit(){
    //   printf("onRunINit\n");
    //   GPUTransfer::resizeCellData();
    //   GPUTransfer::fillCellData();
    //   GPUTransfer::onDecompose();
    // }

    void GPUTransfer::resizeCellData(){
      time = timeIntegrate.getElapsedTime();
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      GPUStorage->numberLocalCells = system.storage->getLocalCells().size();
      timeResizeCellData += timeIntegrate.getElapsedTime() - time;

      time = timeIntegrate.getElapsedTime();
      GPUStorage->resizeCellData();
      timeGResizeCellData += timeIntegrate.getElapsedTime() - time;

      // delete mortonMapping;
      // mortonMapping = NULL;
      cResizeCellData++;
    }

    void GPUTransfer::fillCellData(){
      time = timeIntegrate.getElapsedTime(); // timer
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      CellList localCells = system.storage->getLocalCells();
      int nLocalCells = localCells.size();
      int dim;
      // Morton mapping
      if(mortonSorting){
        dim = ceil(cbrt(nLocalCells));
         
        float base2 = log2((float)dim);
        int nearest2Power = pow(2, ceil(base2));
        int sizeMM = pow(nearest2Power, 3);

        // int sizeMM = pow(pow(2,ceil(log2((float)dim))),3);
        mortonMapping.resize(sizeMM);
        std::fill (mortonMapping.begin(), mortonMapping.end(), -1);
      }
      

      bool realCell;
      int3 mappedPos;
      for(unsigned int i = 0; i < nLocalCells; ++i) {
        if(mortonSorting){
          uint_fast32_t* to3;
          to3 = GPUTransfer::to3D(i, dim, dim, dim);
          uint_fast64_t newPos = libmorton::morton3D_64_encode(to3[0], to3[1], to3[2]);
          mortonMapping[newPos] = i;
        }

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
      }

      if(mortonSorting){
        std::remove_if(mortonMapping.begin(), mortonMapping.end(), [](int i){return i==-1;});
        mortonMapping.resize(nLocalCells);
      }
      timeFillCellData += timeIntegrate.getElapsedTime() - time;

      time = timeIntegrate.getElapsedTime();
      GPUStorage->h2dCellData();
      timeGH2dCellData += timeIntegrate.getElapsedTime() - time;

      cFillCellData++;
    }
    // Cell Data end

    // On decompose, resize Particle Arrays and fill with statics
    //
    void GPUTransfer::onDecompose(){
      GPUTransfer::resizeParticleData();
      GPUTransfer::fillParticleStatics();
      GPUTransfer::fillParticleVars(); // can be uncommented to show correct Step 0 values.
      cOnDecompose++;
    }

    void GPUTransfer::resizeParticleData(){
      time = timeIntegrate.getElapsedTime();
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      GPUStorage->numberLocalParticles = system.storage->getNLocalParticles();
      timeResizeParticleData += timeIntegrate.getElapsedTime() - time;

      time = timeIntegrate.getElapsedTime();
      GPUStorage->resizeParticleData();
      timeGResizeParticleData += timeIntegrate.getElapsedTime() - time;
      
      cResizeParticleData++;
    }

    void GPUTransfer::fillParticleStatics(){
      time = timeIntegrate.getElapsedTime();
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();   
      unsigned int counterParticles = 0;
      int max_n_nb = 0;
      bool realParticle;
      int n_cell_pt;
      int i;
      for(int ii = 0; ii < localCells.size(); ++ii) {
        i = mortonSorting? mortonMapping[ii] : ii;

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
      timeFillParticleStatics += timeIntegrate.getElapsedTime() - time;

      time = timeIntegrate.getElapsedTime();
      GPUStorage->h2dParticleStatics();
      timeGH2dParticleStatics += timeIntegrate.getElapsedTime() - time;

      cFillParticleStatics++;
    }
    // On decompose end

    // Copy particle Pos in each time step
    void GPUTransfer::fillParticleVars(){
      time = timeIntegrate.getElapsedTime();
      System& system = getSystemRef();
      CellList localCells = system.storage->getLocalCells();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();

      unsigned int counterParticles = 0;
      int i;
      for(unsigned int ii = 0; ii < localCells.size(); ++ii) {
        i = mortonSorting? mortonMapping[ii] : ii;

        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          Real3D pos = localCells[i]->particles[j].getPos();
          GPUStorage->h_pos[counterParticles] = make_realG3(pos.at(0), pos.at(1), pos.at(2));
          counterParticles++;
        }
      }
      timeFillParticleVars += timeIntegrate.getElapsedTime() - time;

      time = timeIntegrate.getElapsedTime();
      GPUStorage->h2dParticleVars();
      timeGH2dParticleVars += timeIntegrate.getElapsedTime() - time;

      cFillParticleVars++;
    }

    // Copy forces back in each time step
    void GPUTransfer::getParticleForces(){
      System& system = getSystemRef();
      StorageGPU* GPUStorage = system.storage->getGPUstorage();
      CellList localCells = system.storage->getLocalCells();

      time = timeIntegrate.getElapsedTime();
      GPUStorage->d2hParticleForces();
      timeGD2hParticleForces += timeIntegrate.getElapsedTime() - time;
      
      time = timeIntegrate.getElapsedTime();
      unsigned int counterParticles = 0;
      int i;
      for(int ii = 0; ii < localCells.size(); ++ii) {
        i = mortonSorting? mortonMapping[ii] : ii;

        for(unsigned int j = 0; j < localCells[i]->particles.size(); ++j){
          realG3 force3 = GPUStorage->h_force[counterParticles];
          Real3D force3D(force3.x, force3.y, force3.z);
          Particle &p = localCells[i]->particles[j];
          p.force() += force3D;
          counterParticles++;
        }
      }
      timeGetParticleForces += timeIntegrate.getElapsedTime() - time;

      cGetParticleForces++;
    }

    uint_fast32_t* GPUTransfer::to3D(uint_fast32_t idx, int xMax, int yMax, int zMax){
      uint_fast32_t z = idx / (xMax * yMax);
      idx -= (z * xMax * yMax);
      uint_fast32_t y = idx / xMax;
      uint_fast32_t x = idx % xMax;
      return new uint_fast32_t[3]{ x, y, z }; 
    }

    uint_fast32_t GPUTransfer::to1D( int x, int y, int z , int xMax, int yMax, int zMax ) {
      return (z * xMax * yMax) + (y * xMax) + x;
    }

    void GPUTransfer::printTimers() {
      timeTotal = timeFillParticleVars + timeFillParticleStatics + timeFillCellData + timeGetParticleForces + timeResizeParticleData + timeResizeCellData + timeGResizeParticleData + timeGResizeCellData + timeGH2dParticleStatics + timeGH2dParticleVars + timeGH2dCellData + timeGD2hParticleForces;
      // +timeGPUinit;

      printf("Time for GPU init:        %.3fs\n", timeGPUinit);
      printf("Total GPUTransfer Time:   %.3fs\n", timeTotal);
      printf("timeFillParticleVars:     %.3f s, %.3f %%\n", timeFillParticleVars, 100 * timeFillParticleVars / timeTotal);
      printf("timeGH2dParticleVars:     %.3f s, %.3f %%\n", timeGH2dParticleVars, 100 * timeGH2dParticleVars / timeTotal);

      printf("timeFillParticleStatics:  %.3f s, %.3f %%\n", timeFillParticleStatics, 100 * timeFillParticleStatics / timeTotal);
      printf("timeGH2dParticleStatics:  %.3f s, %.3f %%\n", timeGH2dParticleStatics, 100 * timeGH2dParticleStatics / timeTotal);

      printf("timeFillCellData:         %.3f s, %.3f %%\n", timeFillCellData, 100 * timeFillCellData / timeTotal);
      printf("timeGH2dCellData:         %.3f s, %.3f %%\n", timeGH2dCellData, 100 * timeGH2dCellData / timeTotal);

      printf("timeGetParticleForces:    %.3f s, %.3f %%\n", timeGetParticleForces, 100 * timeGetParticleForces / timeTotal);
      printf("timeGD2hParticleForces:   %.3f s, %.3f %%\n", timeGD2hParticleForces, 100 * timeGD2hParticleForces / timeTotal);

      printf("timeResizeParticleData:   %.3f s, %.3f %%\n", timeResizeParticleData, 100 * timeResizeParticleData / timeTotal);
      printf("timeGResizeParticleData:  %.3f s, %.3f %%\n", timeGResizeParticleData, 100 * timeGResizeParticleData / timeTotal);

      printf("timeResizeCellData:       %.3f s, %.3f %%\n", timeResizeCellData, 100 * timeResizeCellData / timeTotal);
      printf("timeGResizeCellData:      %.3f s, %.3f %%\n", timeGResizeCellData, 100 * timeGResizeCellData / timeTotal);
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
        .def("printTimers", &GPUTransfer::printTimers)
        .def("disableSorting", &GPUTransfer::disableSorting)
        .def("enableSorting", &GPUTransfer::enableSorting)
        ;
    }

  }
}
