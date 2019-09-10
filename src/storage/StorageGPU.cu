#include "StorageGPU.cuh"
#include <stdio.h>
#include <cuda.h>

#define CUERR { \
    cudaError_t cudaerr; \
    if ((cudaerr = cudaGetLastError()) != cudaSuccess){ \
        printf("CUDA ERROR: \"%s\" at LINE %d.\n", cudaGetErrorString(cudaerr), __LINE__); \
    } \
}

__global__ void gpuTest(int N, double *d_px, double *d_py, double *d_pz){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("%d\n", idx);
    if(idx < 10) {
        //printf("Particle: %d, x: %f, y: %f, z: %f\n", idx, d_px[idx], d_py[idx], d_pz[idx]);
    } 

}

__global__ void gpuTest2(int M, int *d_cellOffsets, int *d_numberCellNeighbors){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < M){
        printf("Cell: %d, Offset: %d, #Neighbors: %d\n", idx, d_cellOffsets[idx], d_numberCellNeighbors[idx]);
    }
}

void gpu_resizeParticleData(    int N, 
                                double **d_px, 
                                double **d_py, 
                                double **d_pz, 
                                int **d_type, 
                                double **d_mass, 
                                double **d_drift,
                                double **d_fx, 
                                double **d_fy, 
                                double **d_fz){
    int numBytesD = N * sizeof(double);
    int numBytesI = N * sizeof(int);

    if(*d_px != 0) cudaFree(*d_px);                                                                     CUERR
    if(*d_py != 0) cudaFree(*d_py);                                                                     CUERR
    if(*d_pz != 0) cudaFree(*d_pz);                                                                     CUERR
    if(*d_type != 0) cudaFree(*d_type);                                                                 CUERR
    if(*d_mass != 0) cudaFree(*d_mass);                                                                 CUERR
    if(*d_drift != 0) cudaFree(*d_drift);                                                               CUERR
    if(*d_fx != 0) cudaFree(*d_fx);                                                                     CUERR
    if(*d_fy != 0) cudaFree(*d_fy);                                                                     CUERR
    if(*d_fz != 0) cudaFree(*d_fz);                                                                     CUERR

    cudaMalloc(d_px, numBytesD);                                                                        CUERR
    cudaMalloc(d_py, numBytesD);                                                                        CUERR
    cudaMalloc(d_pz, numBytesD);                                                                        CUERR
    cudaMalloc(d_type, numBytesI);                                                                      CUERR
    cudaMalloc(d_mass, numBytesD);                                                                      CUERR
    cudaMalloc(d_drift, numBytesD);                                                                     CUERR
    cudaMalloc(d_fx, numBytesD);                                                                        CUERR
    cudaMalloc(d_fy, numBytesD);                                                                        CUERR
    cudaMalloc(d_fz, numBytesD);                                                                        CUERR
    cudaMemset(*d_fx, 0, N);                                                                            CUERR
    cudaMemset(*d_fy, 0, N);                                                                            CUERR
    cudaMemset(*d_fz, 0, N);                                                                            CUERR
    
}

void gpu_h2dCellData(   int M, 
                        int **d_cellOffsets, 
                        int **d_numberCellNeighbors, 
                        int *h_cellOffsets, 
                        int *h_numberCellNeighbors) {
    int numBytesI = M * sizeof(int);
    cudaMemcpy(*d_cellOffsets, h_cellOffsets, numBytesI, cudaMemcpyHostToDevice);                       CUERR
    cudaMemcpy(*d_numberCellNeighbors, h_numberCellNeighbors, numBytesI, cudaMemcpyHostToDevice);       CUERR

    //gpuTest2<<<1,128>>>(M, *d_cellOffsets, *d_numberCellNeighbors);
}

void gpu_resizeCellData(  int M,
                            int **d_cellOffsets,
                            int **d_numberCellNeighbors) {

    int numBytes = M * sizeof(int);

    if(*d_cellOffsets != 0 && *d_numberCellNeighbors != 0){
        cudaFree(*d_cellOffsets);                                                                       CUERR
        cudaFree(*d_numberCellNeighbors);                                                               CUERR
    }

    cudaMalloc(d_cellOffsets, numBytes);                                                                CUERR
    cudaMalloc(d_numberCellNeighbors, numBytes);                                                        CUERR

}

void gpu_h2dParticleStatics(    int N,
                                double *h_drift,
                                double **d_drift,
                                double *h_mass,
                                double **d_mass,
                                int *h_type,
                                int **d_type){

    cudaMemcpy(*d_drift, h_drift, N * sizeof(double), cudaMemcpyHostToDevice);                          CUERR
    cudaMemcpy(*d_mass,  h_mass,  N * sizeof(double), cudaMemcpyHostToDevice);                          CUERR
    cudaMemcpy(*d_type,  h_type,  N * sizeof(int),    cudaMemcpyHostToDevice);                          CUERR
}

void gpu_h2dParticleVars(   int N,
                            double *h_px,
                            double **d_px,
                            double *h_py,
                            double **d_py,
                            double *h_pz,
                            double **d_pz){
    cudaMemcpy(*d_px, h_px, sizeof(double) * N, cudaMemcpyHostToDevice);                                CUERR
    cudaMemcpy(*d_py, h_py, sizeof(double) * N, cudaMemcpyHostToDevice);                                CUERR
    cudaMemcpy(*d_pz, h_pz, sizeof(double) * N, cudaMemcpyHostToDevice);                                CUERR
    
    //gpuTest<<<1, 128>>>(N, *d_px, *d_py, *d_pz);                                                        CUERR
    //cudaDeviceSynchronize();                                                                            CUERR

}
void gpu_d2hParticleForces( int N,
                            double **d_fx,
                            double *h_fx,
                            double **d_fy,
                            double *h_fy,
                            double **d_fz,
                            double *h_fz){
    cudaMemcpy(h_fx, *d_fx, sizeof(double) * N, cudaMemcpyDeviceToHost);                                CUERR
    cudaMemcpy(h_fy, *d_fy, sizeof(double) * N, cudaMemcpyDeviceToHost);                                CUERR
    cudaMemcpy(h_fz, *d_fz, sizeof(double) * N, cudaMemcpyDeviceToHost);                                CUERR

}
