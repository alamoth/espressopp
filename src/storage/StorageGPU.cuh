#ifndef __STORAGE_GPU_CUH
#define __STORAGE_GPU_CUH
#include <cuda_runtime.h>

void gpu_resizeParticleData(    int N, 
                                double3 **d_pos,
                                //double **d_px, 
                                //double **d_py, 
                                //double **d_pz, 
                                int **d_type, 
                                double **d_mass, 
                                double **d_drift,
                                double3 **d_force
                                //double **d_fx, 
                                //double **d_fy, 
                                //double **d_fz
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
                            //double *h_px,
                            //double **d_px,
                            //double *h_py,
                            //double **d_py,
                            //double *h_pz,
                            //double **d_pz
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
                            //double **d_fx,
                            //double *h_fx,
                            //double **d_fy,
                            //double *h_fy,
                            //double **d_fz,
                            //double *h_fz
                        );
                           
#endif                            