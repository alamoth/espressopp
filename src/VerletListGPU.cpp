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
#include "VerletListGPU.hpp"
#include "Real3D.hpp"
#include "Particle.hpp"
#include "Cell.hpp"
#include "System.hpp"
#include "storage/Storage.hpp"
#include "bc/BC.hpp"
#include "VerletListGPU.cuh"
#include <cuda_runtime.h>
#include "esutil/CudaHelper.cuh"

#define CUERR { \
  cudaError_t cudaerr; \
  if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
      printf("CUDA ERROR: \"%s\" in File %s at LINE %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__); \
  } \
}
// #include "iterator/CellListAllPairsIterator.hpp"
// #include <stdio.h>
// #include <cuda.h>
#define my_delete(x) {delete x; x = 0;}
namespace espressopp {

  // using namespace espressopp::iterator;

  LOG4ESPP_LOGGER(VerletListGPU::theLogger, "VerletListGPU");

/*-------------------------------------------------------------*/

  // cut is a cutoff (without skin)
  VerletListGPU::VerletListGPU(shared_ptr<System> system, real _cut, bool rebuildVL) : SystemAccess(system)
  {
    LOG4ESPP_INFO(theLogger, "construct VerletListGPU, cut = " << _cut);
  
    if (!system->storage) {
       throw std::runtime_error("system has no storage");
    }

    cut = _cut;
    cutVerlet = cut + system -> getSkin();
    cutsq = cutVerlet * cutVerlet;
    builds = 0;
    vlFactor = 4; //idk, i literally got no idea what to put there

    time = 0;
    timeGPUrebuild = 0;
    timeIntegrate.reset();

    if (rebuildVL) rebuild(); // not called if exclutions are provided

  
    // make a connection to System to invoke rebuild on resort
    // connectionResort = system->storage->aftAftCellAdjust.connect(
    //     boost::bind(&VerletListGPU::rebuild, this));    
    connectionResort2 = system->storage->vlParticlesChanged.connect(
        boost::bind(&VerletListGPU::rebuild, this));
  }
  
  real VerletListGPU::getVerletCutoff(){
    return cutVerlet;
  }
  
  void VerletListGPU::connect()
  {

  // make a connection to System to invoke rebuild on resort
  // connectionResort = getSystem()->storage->aftAftCellAdjust.connect(
  //     boost::bind(&VerletListGPU::rebuild, this));
  connectionResort2 = getSystem()->storage->vlParticlesChanged.connect(
      boost::bind(&VerletListGPU::rebuild, this));
  }

  void VerletListGPU::disconnect()
  {

  // disconnect from System to avoid rebuild on resort
  connectionResort.disconnect();
  }

  /*-------------------------------------------------------------*/
  
  void VerletListGPU::rebuild()
  {
    time = timeIntegrate.getElapsedTime();

    cutVerlet = cut + getSystem() -> getSkin();
    System& system = getSystemRef();
    StorageGPU* GS = system.storage->getGPUstorage();
    cutsq = cutVerlet * cutVerlet;
    max_n_nb = GS->max_n_nb * vlFactor;

    int oldSizeVl = sizeVl;
    int oldNpart = n_pt;
    n_pt = GS->numberLocalParticles;
    sizeVl = max_n_nb * n_pt;

    if(sizeVl > oldSizeVl){
      cudaFree(d_vlPairs); CUERR
      
      my_delete(vlPairs);
      
      vlPairs = new int[sizeVl];
      cudaMalloc((void**)&d_vlPairs, sizeof(int) * sizeVl); CUERR
      
    } else{
      sizeVl = oldSizeVl;
    }
    
    if(n_pt > oldNpart){
      cudaFree(d_n_nb); CUERR
      my_delete(n_nb);
      n_nb = new int[n_pt];
      cudaMalloc((void**)&d_n_nb, sizeof(int) * n_pt); CUERR
    }

    cudaMemset(d_vlPairs, -1, sizeof(int) * sizeVl); CUERR
    cudaMemset(d_n_nb, -1, sizeof(int) * n_pt); CUERR 

    verletListBuildDriver(GS, n_pt, cutsq, d_vlPairs, d_n_nb, max_n_nb);
    builds++;

    timeGPUrebuild += timeIntegrate.getElapsedTime() - time;
  }
  
  int VerletListGPU::totalSize() const
  {
    System& system = getSystemRef();
    int size = localSize();
    int allsize;
  
    mpi::all_reduce(*system.comm, size, allsize, std::plus<int>());
    return allsize;
  }

  int VerletListGPU::localSize() const
  {
    return sizeVl;
  }

  bool VerletListGPU::exclude(longint pid1, longint pid2) {

      exList.insert(std::make_pair(pid1, pid2));
      return true;
  }
  

  /*-------------------------------------------------------------*/
  
  VerletListGPU::~VerletListGPU()
  {
    LOG4ESPP_INFO(theLogger, "~VerletListGPU");
  
    if (!connectionResort.connected()) {
      connectionResort.disconnect();
    }
  }
  
  /****************************************************
  ** REGISTRATION WITH PYTHON
  ****************************************************/
  
  void VerletListGPU::registerPython() {
    using namespace espressopp::python;

    bool (VerletListGPU::*pyExclude)(longint pid1, longint pid2)
          = &VerletListGPU::exclude;


    class_<VerletListGPU, shared_ptr<VerletListGPU> >
      ("VerletListGPU", init< shared_ptr<System>, real, bool >())
      .add_property("system", &SystemAccess::getSystem)
      .add_property("builds", &VerletListGPU::getBuilds, &VerletListGPU::setBuilds)
      .def("totalSize", &VerletListGPU::totalSize)
      .def("localSize", &VerletListGPU::localSize)
      .def("exclude", pyExclude)
      .def("rebuild", &VerletListGPU::rebuild)
      .def("connect", &VerletListGPU::connect)
      .def("disconnect", &VerletListGPU::disconnect)
      .def("getGPUtimer", &VerletListGPU::getRebuildTime)
      .def("getVerletCutoff", &VerletListGPU::getVerletCutoff)
      .def("setVLfactor", &VerletListGPU::setVLfactor)
      .def("getVLfactor", &VerletListGPU::getVLfactor)
      ;
  }

}
