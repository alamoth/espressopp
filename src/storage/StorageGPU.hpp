
#ifndef _STORAGE_GPU_HPP
#define _STORAGE_GPU_HPP
#include "StorageGPU.cuh"
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>



class StorageGPU {
    public:
        StorageGPU() {};
        ~StorageGPU() {};
        
        int numberLocalParticles = 0;
        int numberLocalCells = 0;

        double3 *d_pos = 0;
        double3 *h_pos = 0;
        double3 *um_pos = 0;

        double3 *d_force = 0;
        double3 *h_force = 0;
        double3 *um_force = 0;

        int *h_id = 0;
        int *d_id = 0;
        int *um_id = 0;

        int *h_cellId = 0;
        int *d_cellId = 0;
        int *um_cellId = 0;

        int *h_type = 0;
        int *d_type = 0;
        int *um_type = 0;

        double *h_mass  = 0; 
        double *d_mass = 0;
        double *um_mass = 0;

        double *h_drift = 0;
        double *d_drift = 0;
        double *um_drift = 0;

        bool *h_real = 0;
        bool *d_real = 0;
        bool *um_real = 0;

        int *h_cellOffsets = 0; 
        int *d_cellOffsets = 0;
        int *um_cellOffsets = 0;

        int *h_particlesCell = 0;
        int *d_particlesCell = 0;
        int *um_particlesCell = 0;

        int *h_cellNeighbors = 0;
        int *d_cellNeighbors = 0;

        void resizeParticleData();
        void resizeCellData();

        void h2dParticleStatics();
        void h2dParticleVars();
        void h2dCellData();
        void d2hParticleForces();

    protected:

};

#endif