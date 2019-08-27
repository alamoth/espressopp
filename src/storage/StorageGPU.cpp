#include "StorageGPU.hpp"
#include "StorageGPU.cuh"
#include "types.hpp"
#include "System.hpp"

namespace espressopp{
  
    //StorageGPU::StorageGPU(shared_ptr<System> system) : integrator::Extension(system){}

    StorageGPU::~StorageGPU(){}

    void StorageGPU::h2dParticleVars(){
        printf("duud\n");
        System& system = getSystemRef();
    }

    void StorageGPU::connect(){}
    
    void StorageGPU::disconnect(){}
    
    void StorageGPU::h2dParticleStatics(){}

    void StorageGPU::d2hParticleForce(){}
}
