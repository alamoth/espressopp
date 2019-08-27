/*
#ifdef UpdateForcesGPU_cu
#define UpdateForcesGPU_cu
#include "UpdateForcesGPU.cuh"

//#include "VelocityVerlet.hpp"
//#include "python.hpp"
//#include "VelocityVerlet.hpp"
//#include <iomanip>
//#include "iterator/CellListIterator.hpp"
//#include "interaction/Interaction.hpp"
//#include "interaction/Potential.hpp"
//#include "System.hpp"
//#include "Cell.hpp"
//#include <esutil/ESPPIterator.hpp>

//#include "mpi.hpp"
//#include "SystemAccess.hpp"

#endif
#include <stdio.h>

__global__ void gpuTest() {
	int i = threadIdx.x;
	printf("%d ", i);
}

namespace espressopp{

	double *h_forces, *d_forces;
	double *h_positions, *d_positions; // double or single pointer
	//uint8_t *h_adInfo, *d_adInfo;

void UpdateForcesGPU::h2dParticlePosition(){

	/*longint nRealParticles;
	longint nGhostParticles;
	longint N;
	unsinged bytesN;
	
	// Could also get localCells and check each cell for # of neighbors
	nRealParticles = storage::getNRealParticles();
	nGhostParticles = storage::getNGhostParticles();

	N = nRealParticles + nGhostParticles;
	bytesN = N * sizeof(double);

	// Allocate memory for particle data
	

	h_forces = malloc(bytesN));
	memset(h_forces, 0, bytesN);

	cudaMalloc(&d_forces, bytesN);
	cudaMemset(d_forces, 0, bytesN);

	h_positions = malloc(bytesN * 3); // xyz
	cudaMalloc(&d_positions, bytesN * 3);

	h_adInfo = malloc(sizeof(uint8_t) * N);
	cudaMalloc(&d_adInfo, sizeof(uint8_t) * N);

	printf("CellList size: %d", realCells->size());

	// TODO:
	// Store ID of cell somehow, either by adding info to each particle or additional array
	// with #particles in each cell
	for(CellListIterator cit(realCells); !cit.isDone(); ++cit){
		 
	}

	cudaMemcpy(d_positions, h_positions, bytesN * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_adInfo, h_adInfo, sizeof(uint8_t) * N, cudaMemcpyHostToDevice);
	



	//System& system = getSystemRef();
	//storage::Storage& storage = *system.storage;
	//printf("Number of real particles: %ld", storage.getNRealParticles());
	//printf("Number of Local particles: %ld", storage.getNLocalParticles());
	//printf("Number of Ghost particles: %ld", storage.getNGhostParticles());
	//printf("Number of Adress particles: %ld", storage.getNAdressParticles());
}

void UpdateForcesGPU::gpuForceCalculation(){

}


void UpdateForcesGPU::h2dStaticParticleInfo(){

}

*/