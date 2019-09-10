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

            int numberParticles = 0;
            int numberCells = 0;

            double *d_px = 0;
            double *d_py = 0;
            double *d_pz = 0;

            double *h_px = 0;
            double *h_py = 0;
            double *h_pz = 0;

            double *d_fx = 0;
            double *d_fy = 0;
            double *d_fz = 0;

            double *h_fx = 0;
            double *h_fy = 0;
            double *h_fz = 0;

            int *h_type = 0;
            int *d_type = 0;

            double *h_mass  = 0; 
            double *d_mass = 0;

            double *h_drift = 0;
            double *d_drift = 0;

            int *h_cellOffsets = 0; 
            int *d_cellOffsets = 0;


            int *h_numberCellNeighbors = 0;
            int *d_numberCellNeighbors = 0;

            void resizeParticleData();
            void resizeCellData();

            void h2dParticleStatics();
            void h2dParticleVars();
            void h2dCellData();
            void d2hParticleForces();
            void freeParticleVars();
            void initNullPtr();

        protected:
            //shared_ptr<MDIntegrator> integrator; // this is needed for signal connection
    };

#endif