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
#ifndef _VERLETLISTGPU_HPP
#define _VERLETLISTGPU_HPP

#include "log4espp.hpp"
#include "types.hpp"
#include "python.hpp"
#include "Particle.hpp"
#include "SystemAccess.hpp"
#include "boost/signals2.hpp"
#include "boost/unordered_set.hpp"
#include <cuda_runtime.h>
#include "esutil/Timer.hpp"

namespace espressopp {

/** Class that builds and stores verlet lists.

    ToDo: register at system for rebuild

*/

  class VerletListGPU : public SystemAccess {

  public:

    /** Build a verlet list of all particle pairs in the storage
	whose distance is less than a given cutoff.

	\param system is the system for which the verlet list is built
	\param cut is the cutoff value for the 

    */

    VerletListGPU(shared_ptr< System >, real cut, bool rebuildVL);

    ~VerletListGPU();

    int* getPairs() { return vlPairs; }

    int* getPairsGPU() { return d_vlPairs;}

    int* getNumNb() { return n_nb; }

    int* getNumNbGPU() { return d_n_nb; }
    
    real getVerletCutoff(); // returns cutoff + skin

    void connect();

    void disconnect();

    void rebuild();

    /** Get the total number of pairs for the Verlet list */
    int totalSize() const;

    //** Get the number of pairs for the local Verlet list */
    int localSize() const;

    /** Add pairs to exclusion list */
    bool exclude(longint pid1, longint pid2);

    /** Get the number of times the Verlet list has been rebuilt */
    int getBuilds() const { return builds; }

    /** Set the number of times the Verlet list has been rebuilt */
    void setBuilds(int _builds) { builds = _builds; }

    double getRebuildTime() const { return timeGPUrebuild; }

    /** Register this class so it can be used from Python. */
    static void registerPython();

  protected:

    void checkPair(Particle &pt1, Particle &pt2);
    int* vlPairs = 0;
    int* d_vlPairs = 0;
    int* n_nb = 0;
    int* d_n_nb = 0;
    int sizeVl = 0;
    int max_n_nb;
    int n_pt;

    esutil::WallTimer timeIntegrate; // timers
    double time;
    double timeGPUrebuild;

    boost::unordered_set<std::pair<longint, longint> > exList; // exclusion list
    
    real cutsq;
    real cut;
    real cutVerlet;
    
    int builds;
    boost::signals2::connection connectionResort, connectionResort2;
    
    static LOG4ESPP_DECL_LOGGER(theLogger);
  };

}

#endif
