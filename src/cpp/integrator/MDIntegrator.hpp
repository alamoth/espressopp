#ifndef _MDINTEGRATOR_HPP
#define _MDINTEGRATOR_HPP

namespace espresso {

  namespace integrator {

    class MDIntegrator {

    protected:

       real timeStep;
       real timeStepSqr;

    public:

       virtual ~MDIntegrator() {}

       virtual void setTimeStep(real _timeStep) { timeStep = _timeStep; timeStepSqr = timeStep * timeStep; }

       virtual void run(int nsteps) = 0;

    };

  }
}

#endif

