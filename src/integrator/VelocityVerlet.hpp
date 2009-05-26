#ifndef _VELOCITY_VERLET
#define _VELOCITY_VERLET

#include <boost/signals2.hpp>

#include "types.hpp"
#include "MDIntegrator.hpp"
#include "particles/Set.hpp"
#include "interaction/Interaction.hpp"
#include "pairs/Set.hpp"
#include "Property.hpp"

namespace espresso {
  namespace integrator {

    class VelocityVerlet: public MDIntegrator {

    private:

      boost::shared_ptr<particles::Set> particles;
      // boost::shared_ptr<Stoarge> storage;

      boost::shared_ptr< Property<Real3D> > position;
      boost::shared_ptr< Property<Real3D> > velocity;
      boost::shared_ptr< Property<Real3D> > force;

      struct ForceEvaluation {
        boost::shared_ptr<interaction::Interaction> interaction;
        boost::shared_ptr<pairs::Set> pairs;
        ForceEvaluation(boost::shared_ptr<interaction::Interaction> _interaction,
                        boost::shared_ptr<pairs::Set> _pairs)
          : interaction(_interaction), pairs(_pairs) {}
      };

      std::vector<ForceEvaluation> forceEvaluations;

    public:

      static void registerPython();

      VelocityVerlet(real _timeStep);

      VelocityVerlet(boost::shared_ptr<particles::Set> _particles, 
                     boost::shared_ptr< Property<Real3D> > _position,
                     boost::shared_ptr< Property<Real3D> > _velocity,
                     boost::shared_ptr< Property<Real3D> > _force);
      
      void addForce(boost::shared_ptr<interaction::Interaction> interaction, 
                    boost::shared_ptr<pairs::Set> pairs);

      boost::shared_ptr<particles::Set> getSet() { return particles; }
  
      /** All routines that should be called after stepA in one integration 
          step must connect to this signal.
      */

      boost::signals2::signal0<void> postStepA;

      /** All routines that should be called after stepB in one integration 
          step must connect to this signal.
      */

      boost::signals2::signal1<void, int> postStepB;

      virtual void run(int timesteps); 

      virtual ~VelocityVerlet();

   };

  }
}

#endif
