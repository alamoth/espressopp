#ifndef _STORAGE_GPU_HPP
#define _STORAGE_GPU_HPP
//#include "SystemAccess.hpp"
//#include "python.hpp"
//#include "types.hpp"
//#include "integrator/Extension.hpp"
//#include <boost/signals2.hpp>
//#include "SystemAccess.hpp"



namespace espressopp{
    
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

            double *d_x;
            double *d_y;
            double *d_z;

            double *h_x;
            double *h_y;
            double *h_z;

            unsigned *h_type;
            unsigned *d_type;

            double *h_mass, *d_mass;

            double *h_drift, *d_drift;

            unsigned *cellOffsets;

            unsigned *numberCellNeighbors;

        protected:
            //shared_ptr<MDIntegrator> integrator; // this is needed for signal connection
    };
}

#endif