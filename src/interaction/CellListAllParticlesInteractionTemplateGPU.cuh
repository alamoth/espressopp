#ifndef _INTERACTION_CELLLISTALLPARTICLESINTERACTIONTEMPLATEGPU_CUH
#define _INTERACTION_CELLLISTALLPARTICLESINTERACTIONTEMPLATEGPU_CUH

#include "cuda.h"
#include "stdio.h"
//#include "storage/Storage.hpp"
//#include "types.hpp"

#define CUERR { \
    cudaError_t cudaerr; \
    if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
        printf("CUDA ERROR: \"%s\" in File %s at LINE %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__); \
    } \
}



namespace espressopp{
namespace interaction{
    class LennardJonesGPU;

    class gpu_CellListAllParticlesInteractionTemplateGPU{
        public:
        LennardJonesGPU* _ljgpu;
        gpu_CellListAllParticlesInteractionTemplateGPU(){};

        /*
        inline void gpu_addPotential (Potential potential, int x, int y){
            cudaMalloc(&d_potentials, sizeof(Potential));                                           CUERR
            cudaMemcpy(d_potentials, &potential, sizeof(Potential), cudaMemcpyHostToDevice);        CUERR
            printf("Shift: %f\n", potential.getSigma());
            KernelTest<Potential><<<1,1>>>(d_potentials);
        } */  
        //template <class _Potential> 
        void testF(){};
        template<typename Potential> void
            gpuTest(Potential *d_potentials);

        //shared_ptr< storage::Storage > storage;
        //Potential *d_potentials;

    };
}
}
#endif