//#include "cuda.h"
#include "cuda_runtime.h"
#include "CellListAllParticlesInteractionTemplateGPU.cuh"
//#include <stdio.h>
//#include "LennardJonesGPU.hpp"

/*
namespace espressopp{


    template<typename _Potential>
    void gpu_CellListAllParticlesInteractionTemplateGPU::kernelWrapper(){
        KernelTest<_Potential><<<1,1>>>(d_potentials);
    }
    
    template class gpu_CellListAllParticlesInteractionTemplateGPU<LennardJonesGPU>;
}
*/


namespace espressopp{
namespace interaction{
    template<typename Potential>
    __global__
    void KernelTest(Potential *potentials){
        Potential* test = potentials;
        printf("HiGPU: %f\n", test->getSigma());

}

    template<typename Potential> void 
    gpu_CellListAllParticlesInteractionTemplateGPU::gpuTest(Potential *d_potentials){
        //printf("Sigma: %f\n", d_potentials->getSigma());
        //KernelTest<Potential><<<1,1>>>(d_potentials);
    }
    template void gpu_CellListAllParticlesInteractionTemplateGPU::gpuTest<espressopp::interaction::LennardJonesGPU>(espressopp::interaction::LennardJonesGPU *d_potentials);


}
}
