#include "StorageGPU.cuh"
#include <stdio.h>
#include <cuda.h>

#define CUERR { \
    cudaError_t cudaerr; \
    if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
        printf("CUDA ERROR: \"%s\" in File %s at LINE %d.\n", cudaGetErrorString(cudaerr), __FILE__, __LINE__); \
    } \
}

void gpu_resizeParticleData(    int N, 
                                int **d_cellId,
                                int **d_id,
                                int **d_type, 
                                realG **d_drift,
                                realG **d_mass, 
                                realG3 **d_pos,
                                realG3 **d_force,
                                bool **d_real
                            ){
    int numBytesD = N * sizeof(realG);
    int numBytesI = N * sizeof(int);

    if(*d_cellId != 0) cudaFree(*d_cellId);                                                             CUERR
    if(*d_id != 0) cudaFree(*d_id);                                                                     CUERR
    if(*d_type != 0) cudaFree(*d_type);                                                                 CUERR
    if(*d_drift != 0) cudaFree(*d_drift);                                                               CUERR
    if(*d_mass != 0) cudaFree(*d_mass);                                                                 CUERR
    if(*d_pos != 0) cudaFree(*d_pos);                                                                   CUERR
    if(*d_force != 0) cudaFree(*d_force);                                                               CUERR
    if(*d_real != 0) cudaFree(*d_real);                                                                 CUERR

    cudaMalloc(d_cellId, numBytesI);                                                                    CUERR 
    cudaMalloc(d_id, numBytesI);                                                                        CUERR 
    cudaMalloc(d_type, numBytesI);                                                                      CUERR
    cudaMalloc(d_pos,   sizeof(realG3) * N);                                                           CUERR
    cudaMalloc(d_force, sizeof(realG3) * N);                                                           CUERR
    cudaMalloc(d_mass, numBytesD);                                                                      CUERR
    cudaMalloc(d_drift, numBytesD);                                                                     CUERR
    cudaMalloc(d_real, sizeof(bool) * N);                                                               CUERR

    cudaMemset(*d_force, 0, sizeof(realG3) * N);                                                       CUERR
    
}

void gpu_h2dCellData(   int M, 
                        int *h_cellNeighbors,
                        int **d_cellNeighbors
                    ) {
    int numBytesI = M * sizeof(int);
    cudaMemcpy(*d_cellNeighbors, h_cellNeighbors, numBytesI * 27, cudaMemcpyHostToDevice);              CUERR
}

void gpu_resizeCellData(    int M,
                            int **d_cellOffsets,
                            int **d_particlesCell,
                            int **d_cellNeighbors) {

    int numBytes = M * sizeof(int);

    if(*d_cellOffsets != 0 && *d_particlesCell != 0){
        cudaFree(*d_cellOffsets);                                                                       CUERR
        cudaFree(*d_particlesCell);                                                                     CUERR
        cudaFree(*d_cellNeighbors);                                                                     CUERR
    }

    cudaMalloc(d_cellOffsets, numBytes);                                                                CUERR
    cudaMalloc(d_particlesCell, numBytes);                                                              CUERR
    cudaMalloc(d_cellNeighbors, numBytes * 27);                                                         CUERR

}

void gpu_h2dParticleStatics(    int nLocalParticles,
                                int nLocalCells,
                                int *h_cellId,
                                int **d_cellId,
                                int *h_id,
                                int **d_id,
                                int *h_type,
                                int **d_type,
                                realG *h_drift,
                                realG **d_drift,
                                realG *h_mass,
                                realG **d_mass,
                                bool *h_real,
                                bool **d_real,
                                int *h_cellOffsets,
                                int **d_cellOffsets,
                                int *h_particlesCell,
                                int **d_particlesCell
                                ){

    cudaMemcpy(*d_cellId,  h_cellId,  nLocalParticles * sizeof(int),cudaMemcpyHostToDevice);                          CUERR
    cudaMemcpy(*d_type,  h_type,  nLocalParticles * sizeof(int),    cudaMemcpyHostToDevice);                          CUERR
    cudaMemcpy(*d_id,  h_id,  nLocalParticles * sizeof(int),    cudaMemcpyHostToDevice);                              CUERR
    cudaMemcpy(*d_drift, h_drift, nLocalParticles * sizeof(realG), cudaMemcpyHostToDevice);                          CUERR
    cudaMemcpy(*d_mass,  h_mass,  nLocalParticles * sizeof(realG), cudaMemcpyHostToDevice);                          CUERR
    cudaMemcpy(*d_real,  h_real,  nLocalParticles * sizeof(bool), cudaMemcpyHostToDevice);                            CUERR
    cudaMemcpy(*d_cellOffsets, h_cellOffsets, nLocalCells * sizeof(int), cudaMemcpyHostToDevice);                       CUERR
    cudaMemcpy(*d_particlesCell, h_particlesCell, nLocalCells * sizeof(int), cudaMemcpyHostToDevice);                   CUERR
}

void gpu_h2dParticleVars(   int N,
                            realG3 *h_pos,
                            realG3 **d_pos
                        ){

    cudaMemcpy(*d_pos, h_pos, sizeof(realG3) * N, cudaMemcpyHostToDevice);                             CUERR

}
void gpu_d2hParticleForces( int N,
                            realG3 *h_force,
                            realG3 **d_force
                        ){
    cudaDeviceSynchronize();  CUERR                        
    cudaMemcpy(h_force, *d_force, sizeof(realG3) * N, cudaMemcpyDeviceToHost);                         CUERR
    cudaMemset(*d_force, 0, sizeof(realG3) * N);                        
}
