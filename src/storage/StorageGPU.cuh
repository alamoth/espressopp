#ifndef __STORAGE_GPU_CUH
#define __STORAGE_GPU_CUH
#include <cuda_runtime.h>

void gpu_resizeParticleData(    int N, 
                                double3 **d_pos,
                                int **d_type, 
                                double **d_mass, 
                                double **d_drift,
                                double3 **d_force
                            );


void gpu_h2dCellData(   int M, 
                        int **d_cellOffsets, 
                        int **d_numberCellNeighbors, 
                        int *h_cellOffsets, 
                        int *h_numberCellNeighbors
                    ); 

void gpu_resizeCellData(  int N,
                            int **d_cellOffsets,
                            int **d_numberCellNeighbors);


void gpu_h2dParticleVars(   int N,
                            double3 *h_pos,
                            double3 **d_pos
                        );

void gpu_h2dParticleStatics(    int N,
                                double *h_drift,
                                double **d_drift,
                                double *h_mass,
                                double **d_mass,
                                int *h_type,
                                int **d_type);

void gpu_d2hParticleForces( int N,
                            double3 *h_force,
                            double3 **d_force
                        );
                           
#endif                            