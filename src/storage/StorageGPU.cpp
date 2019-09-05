#include "StorageGPU.hpp"
#include "StorageGPU.cuh"

//#include "types.hpp"
//#include "System.hpp"
//#include "Cell.hpp"
//#include "Storage.hpp"

  
    //StorageGPU::StorageGPU(shared_ptr<System> system) : integrator::Extension(system){}

    //StorageGPU::~StorageGPU(){}
    

    void StorageGPU::allocateParticleData(){
        //printf("AllocateParticleData\n");
        h_px = new double[numberParticles];
        h_py = new double[numberParticles];
        h_pz = new double[numberParticles];
        h_type = new int[numberParticles];
        h_mass = new double[numberParticles];
        h_drift = new double[numberParticles];
        h_fx = new double[numberParticles];
        h_fy = new double[numberParticles];
        h_fz = new double[numberParticles]; 

        gpu_allocateParticleData(numberParticles, &d_px, &d_py, &d_pz, &d_type, &d_mass, &d_drift, &d_fx, &d_fy, &d_fz);
    }

    void StorageGPU::allocateCellData(){
        h_cellOffsets = new int[numberCells];
        h_numberCellNeighbors = new int[numberCells];
        gpu_allocateCellData(numberCells, &d_cellOffsets, &d_numberCellNeighbors);
    }
    
    void StorageGPU::h2dParticleStatics(){
        gpu_h2dParticleStatics( numberParticles,
                                h_drift,
                                &d_drift,
                                h_mass,
                                &d_mass,
                                h_type,
                                &d_type);
    }

    void StorageGPU::h2dParticleVars(){
        gpu_h2dParticleVars(numberParticles, 
                            h_px,
                            &d_px,
                            h_py,
                            &d_py,
                            h_pz,
                            &d_pz);
    }
    void StorageGPU::d2hParticleForces(){

    }
    
    void StorageGPU::freeParticleVars(){

    }

    void StorageGPU::initNullPtr(){
        
    }

