#ifndef __STORAGE_GPU_CUH
#define __STORAGE_GPU_CUH
#include <cuda_runtime.h>

typedef struct {
    int numberParticles;
    int *cellId;
    int *type;
    double *mass;
    double *drift;
    bool *ghost;
} particleStatics;

typedef struct {
    int numberCells;
    int *cellOffsets;
    int *numberCellNeighbors;
} cellInfo;

void gpu_resizeParticleData(    int N, 
                                int **d_cellId,
                                int **d_id,
                                int **d_type, 
                                double **d_drift,
                                double **d_mass, 
                                double3 **d_pos,
                                double3 **d_force,
                                bool **d_ghost
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
                                int *h_cellId,
                                int **d_cellId,
                                int *h_id,
                                int **d_id,
                                int *h_type,
                                int **d_type,
                                double *h_drift,
                                double **d_drift,
                                double *h_mass,
                                double **d_mass,
                                bool *h_ghost,
                                bool **d_ghost
                                );

void gpu_d2hParticleForces( int N,
                            double3 *h_force,
                            double3 **d_force
                        );
                           
#endif                            