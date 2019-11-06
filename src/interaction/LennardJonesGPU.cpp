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
#include <boost/signals2.hpp>
#include "CellListAllParticlesInteractionTemplateGPU.hpp"
#include "LennardJonesGPU.hpp"
#include "LennardJonesGPU.cuh"

namespace espressopp {
  namespace interaction {

    typedef class CellListAllParticlesInteractionTemplateGPU <LennardJonesGPU, d_LennardJonesGPU> CellListLennardJonesGPU;
   

    //////////////////////////////////////////////////
    // REGISTRATION WITH PYTHON
    //////////////////////////////////////////////////
    void LennardJonesGPU::registerPython() {
      using namespace espressopp::python;

      class_< LennardJonesGPU, bases< Potential > >
    	("interaction_LennardJonesGPU", init< real, real, real>())
      .def(init< real, real, real, real>())
      .add_property("sigma", &LennardJonesGPU::getSigma, &LennardJonesGPU::setSigma)
    	.add_property("epsilon", &LennardJonesGPU::getEpsilon, &LennardJonesGPU::setEpsilon);
      ;

      class_< CellListLennardJonesGPU, bases< Interaction > >
        ("interaction_CellListLennardJonesGPU",	init< shared_ptr< storage::Storage >, shared_ptr<VerletListGPU> >())
        .def("getVerletList", &CellListLennardJonesGPU::getVerletList)
        .def("getPotential", &CellListLennardJonesGPU::getPotentialPtr)
        .def("setPotential", &CellListLennardJonesGPU::setPotential);
	  ;

    }
    
  }
}
