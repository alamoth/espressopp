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
    if(idx < 1) {
        printf("Particle: %d, x: %f, y: %f, z: %f\n", idx, d_px[idx], d_py[idx], d_pz[idx]);
    } 
}

__global__ void gpuTest2(){
    printf("%d\n", threadIdx.x);
}

void gpu_allocateParticleData(  int N, 
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
    cudaMalloc(d_px, numBytesD);                                                                        CUERR
    cudaMalloc(d_py, numBytesD);                                                                        CUERR
    cudaMalloc(d_pz, numBytesD);                                                                        CUERR
    cudaMalloc(d_type, sizeof(int) * N);                                                                CUERR
    cudaMalloc(d_mass, numBytesD);                                                                      CUERR
    cudaMalloc(d_drift, numBytesD);                                                                     CUERR
}

void gpu_allocateCellData(  int N,
                            int **d_cellOffsets,
                            int **d_numberCellNeighbors) {
    printf("AllocateCelldata\n");
    int numBytes = N * sizeof(int);
    cudaMalloc(d_cellOffsets, numBytes);
    cudaMalloc(d_numberCellNeighbors, numBytes);

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
    
    //printf("Before kernel, N= %d\n", N);
    gpuTest<<<1, 128>>>(N, *d_px, *d_py, *d_pz);                                                        CUERR
    //gpuTest2<<<1,128>>>(); CUERR
    cudaDeviceSynchronize();                                                                            CUERR

}
void gpu_d2hParticleForces( int N,
                            double **d_fx,
                            double *h_fx,
                            double **d_fy,
                            double *h_fy,
                            double **d_fz,
                            double *h_fz){

}
