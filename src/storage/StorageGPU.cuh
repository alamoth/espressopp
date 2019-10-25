#define realG double
#define realG3 double4
#define realG4 double4
#define make_realG3 make_double4
#define make_realG4 make_double4

#ifndef __STORAGE_GPU_CUH
#define __STORAGE_GPU_CUH
#include <cuda_runtime.h>

typedef struct {
    int numberParticles;
    int *cellId;
    int *type;
    realG *mass;
    realG *drift;
    bool *real;
} particleStatics;

typedef struct {
    int numberCells;
    int *cellOffsets;
    int *particlesCell;
} cellInfo;

void gpu_resizeParticleData(    int N, 
                                int **d_cellId,
                                int **d_id,
                                int **d_type, 
                                realG **d_drift,
                                realG **d_mass, 
                                realG3 **d_pos,
                                realG3 **d_force,
                                bool **d_real
                            );


void gpu_h2dCellData(   int M, 
                        int *h_cellNeighbors,
                        int **d_cellNeighbors
                    ); 

void gpu_resizeCellData(  int N,
                            int **d_cellOffsets,
                            int **d_particlesCell,
                            int **d_cellNeighbors);


void gpu_h2dParticleVars(   int N,
                            realG3 *h_pos,
                            realG3 **d_pos
                        );

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
                            );

void gpu_d2hParticleForces( int N,
                            realG3 *h_force,
                            realG3 **d_force
                        );
                           
#endif                            