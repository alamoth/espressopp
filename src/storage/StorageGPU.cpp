#include "StorageGPU.hpp"
//#include "StorageGPU.cuh"

    
#define my_delete(x) {delete x; x = 0;}

    void StorageGPU::resizeParticleData(){
        my_delete(h_cellId);
        my_delete(h_id);
        my_delete(h_type);
        my_delete(h_mass);
        my_delete(h_drift);
        my_delete(h_real);

        my_delete(h_pos);
        my_delete(h_force);

        h_cellId = new int[numberLocalParticles];
        h_id = new int[numberLocalParticles];
        h_type = new int[numberLocalParticles];
        h_drift = new double[numberLocalParticles];
        h_mass = new double[numberLocalParticles];
        h_real = new bool[numberLocalParticles];

        h_pos = new double3[numberLocalParticles];
        h_force = new double3[numberLocalParticles];

        gpu_resizeParticleData(numberLocalParticles, &d_cellId, &d_id, &d_type, &d_drift, &d_mass, &d_pos, &d_force, &d_real);
    }

    void StorageGPU::resizeCellData(){
        my_delete(h_cellOffsets);
        my_delete(h_particlesCell);
        my_delete(h_cellNeighbors);

        h_cellOffsets = new int[numberLocalCells];
        h_particlesCell = new int[numberLocalCells];
        h_cellNeighbors = new int[numberLocalCells * 27];
        gpu_resizeCellData(numberLocalCells, &d_cellOffsets, &d_particlesCell, &d_cellNeighbors);
    }

    void StorageGPU::h2dCellData(){
        gpu_h2dCellData(    numberLocalCells,
                            &d_cellOffsets,
                            &d_particlesCell,
                            &d_cellNeighbors,
                            h_cellOffsets,
                            h_particlesCell,
                            h_cellNeighbors);
    }
    
    void StorageGPU::h2dParticleStatics(){
        gpu_h2dParticleStatics( numberLocalParticles,
                                h_cellId,
                                &d_cellId,
                                h_id,
                                &d_id,
                                h_type,
                                &d_type,
                                h_drift,
                                &d_drift,
                                h_mass,
                                &d_mass,
                                h_real,
                                &d_real);
    }

    void StorageGPU::h2dParticleVars(){
        gpu_h2dParticleVars(numberLocalParticles, h_pos, &d_pos);
    }

    void StorageGPU::d2hParticleForces(){
        gpu_d2hParticleForces(numberLocalParticles, h_force, &d_force);
    }
