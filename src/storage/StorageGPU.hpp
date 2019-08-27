#ifndef _STORAGE_GPU_HPP
#define _STORAGE_GPU_HPP
#include "SystemAccess.hpp"
#include "python.hpp"
#include "types.hpp"
#include "integrator/Extension.hpp"
#include <boost/signals2.hpp>
#include "SystemAccess.hpp"



namespace espressopp{
    
    class MDIntegrator; //fwd declaration
    //class StorageGPU : public integrator::Extension {
    class StorageGPU : public SystemAccess{
        public:
            StorageGPU(shared_ptr<class espressopp::System> system) : SystemAccess (system) {};
            ~StorageGPU();
            
            void h2dParticleVars();
            void h2dParticleStatics();
            void d2hParticleForce();
            void connect();
            void disconnect();
            boost::signals2::connection _aftInitF;
            boost::signals2::connection _aftCalcF;

        protected:
            shared_ptr<MDIntegrator> integrator; // this is needed for signal connection
    };
}

#endif