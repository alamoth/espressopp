/*
#ifndef UpdateForcesGPU_cuh
#define UpdateForcesGPU_cuh
#include "VelocityVerlet.hpp"
//#include <mpi.hpp>
//#include <boost/thread.hpp>
#include <stdio.h>
//#include "types.hpp"
//#include "MDIntegrator.hpp"
//#include "esutil/Timer.hpp"
//#include <boost/signals2.hpp>
//#include "VelocityVerlet.hpp"
//#include "Cell.hpp"
//#include "storage/Storage.hpp"




//__global__ void gpuTest();

namespace espressopp{
class UpdateForcesGPU{
    public:
    UpdateForcesGPU();
    void h2dParticlePosition();
    void gpuForceCalculation();
    void h2dStaticParticleInfo();


};
}
#endif
*/
