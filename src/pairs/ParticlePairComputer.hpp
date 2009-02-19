#ifndef _PAIRS_PARTICLEPAIRCOMPUTER_HPP
#define _PAIRS_PARTICLEPAIRCOMPUTER_HPP

#include "types.hpp"

#include "particleset/ParticleSet.hpp"

namespace espresso {
  
  namespace pairs {
    /** Abstract class that defines the operator() applied to particle pairs
     */
    template<class ParticleReference>
    class ParticlePairComputerBase {
      
    public:
      /// @name extended function object interface
      //@{
      typedef Real3D             first_argument_type;
      typedef ParticleReference second_argument_type;
      typedef ParticleReference  third_argument_type;
      typedef void                       result_type;
      //@}

      /** Interface of the routine that is applied to particle pairs
	  \param dist: distance vector between the two particles
	  \param p1, p2: references to the two particles

	  Note: The references are necessary if more property data of the particles is
	  needed than only the distance.
      */
      virtual void operator()(const Real3D &dist, 
			      const ParticleReference p1, 
			      const ParticleReference p2) = 0;
    };

    /** Abstract class that defines a function on pairs of particles */
    class ParticlePairComputer: 
      public ParticlePairComputerBase<espresso::particlestorage::ParticleStorage::reference> 
    { };
    
    /** Abstract class that defines a function on pairs of read-only particles */
    class ConstParticlePairComputer:
      public ParticlePairComputerBase<espresso::particlestorage::ParticleStorage::const_reference> 
    {};
  }
}

#endif