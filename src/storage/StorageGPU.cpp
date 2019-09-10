#include "StorageGPU.hpp"
#include "StorageGPU.cuh"

//#include "types.hpp"
//#include "System.hpp"
//#include "Cell.hpp"
//#include "Storage.hpp"

  
    //StorageGPU::StorageGPU(shared_ptr<System> system) : integrator::Extension(system){}

    //StorageGPU::~StorageGPU(){}
    
#define my_delete(x) {delete x; x = 0;}

    void StorageGPU::resizeParticleData(){
        //printf("resizeParticleData\n");
        my_delete(h_px);
        my_delete(h_py);
        my_delete(h_pz);
        my_delete(h_type);
        my_delete(h_mass);
        my_delete(h_drift);
        my_delete(h_fx);
        my_delete(h_fy);
        my_delete(h_px);
        my_delete(h_fz);

        h_px = new double[numberParticles];
        h_py = new double[numberParticles];
        h_pz = new double[numberParticles];
        h_type = new int[numberParticles];
        h_mass = new double[numberParticles];
        h_drift = new double[numberParticles];
        h_fx = new double[numberParticles];
        h_fy = new double[numberParticles];
        h_fz = new double[numberParticles]; 

        gpu_resizeParticleData(numberParticles, &d_px, &d_py, &d_pz, &d_type, &d_mass, &d_drift, &d_fx, &d_fy, &d_fz);
    }

    void StorageGPU::resizeCellData(){
        my_delete(h_cellOffsets);
        my_delete(h_numberCellNeighbors);

        h_cellOffsets = new int[numberCells];
        h_numberCellNeighbors = new int[numberCells];
        
        gpu_resizeCellData(numberCells, &d_cellOffsets, &d_numberCellNeighbors);
    }

    void StorageGPU::h2dCellData(){
        gpu_h2dCellData(    numberCells,
                            &d_cellOffsets,
                            &d_numberCellNeighbors,
                            h_cellOffsets,
                            h_numberCellNeighbors);
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
        gpu_d2hParticleForces(  numberParticles, 
                                &d_fx, 
                                h_fx, 
                                &d_fy, 
                                h_fy, 
                                &d_fz, 
                                h_fz);
    }
    
    void StorageGPU::freeParticleVars(){

    }

    void StorageGPU::initNullPtr(){
        
    }

