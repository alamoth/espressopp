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
    }

    void GPUTransfer::disconnect(){
      //_aftInitF.disconnect();
    }

    void GPUTransfer::connect(){
      // connection to initialisation
  	  //_aftInitF  = integrator->aftInitF.connect( boost::bind(&GPUTransfer::applyForceToAll, this));
    }

    void h2dParticleVars(){

    }
    void h2dParticleStatics(){

    }
    void d2hParticleForces(){
      
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
