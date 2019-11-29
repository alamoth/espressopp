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
#ifndef _INTEGRATOR_GPUTRANSFER_HPP
#define _INTEGRATOR_GPUTRANSFER_HPP

#include "types.hpp"
#include "logging.hpp"
#include "Extension.hpp"
#include "ParticleGroup.hpp"
#include "boost/signals2.hpp"
#include "Real3D.hpp"
#include "Particle.hpp"
#include "esutil/Timer.hpp"

namespace espressopp {
  namespace integrator {

    /** GPUTransfer */

    class GPUTransfer : public Extension {

      public:

        GPUTransfer(shared_ptr< System > _system);

        void onDecompose();
        void onRunInit();
        void onGridInit();

        void fillParticleVars();
        void fillParticleStatics();
        void fillCellData();
        void getParticleForces();

        void resizeParticleData();
        void resizeCellData();

        uint_fast32_t* to3D(uint_fast64_t idx, int xMax, int yMax, int zMax);
        uint_fast64_t  to1D( int x, int y, int z , int xMax, int yMax, int zMax );

        void enableSorting() { mortonSorting = true; }
        void disableSorting() { mortonSorting = false; }

        virtual ~GPUTransfer() {};

        /** Register this class so it can be used from Python. */
        static void registerPython();

      private:
        boost::signals2::connection _aftCalcF, _aftInitF, _onParticlesChanged, _runInit, _aftCellAdjust;
        void connect();
        void disconnect();
        void printTimers();

        std::vector<int> mortonMapping;
        bool mortonSorting = false;

        esutil::WallTimer timeIntegrate; // timers
        double time;
        double timeTotal;       
        double timeFillParticleVars; //
        double timeFillParticleStatics; //
        double timeFillCellData; //
        double timeGetParticleForces; //
        double timeResizeParticleData; //
        double timeResizeCellData; //
        double timeGPUinit;
        
        double timeGResizeParticleData; //
        double timeGResizeCellData; //
        double timeGH2dParticleStatics; //
        double timeGH2dParticleVars; //
        double timeGH2dCellData;  //
        double timeGD2hParticleForces; //

        int cOnDecompose;
        int cOnGridInit;
        int cFillParticleVars;
        int cFillParticleStatics;
        int cFillCellData;
        int cGetParticleForces;
        int cResizeParticleData;
        int cResizeCellData;

        /** Logger */
        static LOG4ESPP_DECL_LOGGER(theLogger);
    };
  }
}

#endif
