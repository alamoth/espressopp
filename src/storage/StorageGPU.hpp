#include "StorageGPU.cuh"
#ifndef _STORAGE_GPU_HPP
#define _STORAGE_GPU_HPP
//#include "SystemAccess.hpp"
//#include "python.hpp"
//#include "types.hpp"
//#include "integrator/Extension.hpp"
//#include <boost/signals2.hpp>
//#include "SystemAccess.hpp"
#include <vector>





    
    //class MDIntegrator; //fwd declaration
    //class StorageGPU : public integrator::Extension {
    class StorageGPU {
        public:
            StorageGPU() {};
            ~StorageGPU() {};
            
            //void h2dParticleVars();
            //void h2dParticleStatics();
            //void d2hParticleForce();
            //void connect();
            //void disconnect();
            //boost::signals2::connection _aftInitF;
            //boost::signals2::connection _aftCalcF;
            //boost::signals2::connection _onParticlesChanged;

            int numberParticles;
            int numberCells;

            double *d_px;
            double *d_py;
            double *d_pz;

            double *h_px;
            double *h_py;
            double *h_pz;

            double *d_fx;
            double *d_fy;
            double *d_fz;

            double *h_fx;
            double *h_fy;
            double *h_fz;

            int *h_type;
            int *d_type;

            double *h_mass, *d_mass;

            double *h_drift, *d_drift;

            int *h_cellOffsets, *d_cellOffsets;

            int *h_numberCellNeighbors, *d_numberCellNeighbors;

            void allocateParticleData();
            void allocateCellData();

            void h2dParticleStatics();
            void h2dParticleVars();
            void d2hParticleForces();
            void freeParticleVars();
            void initNullPtr();

        protected:
            //shared_ptr<MDIntegrator> integrator; // this is needed for signal connection
    };

#endif