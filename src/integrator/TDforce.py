from espresso.esutil import cxxinit
from espresso import pmi

from _espresso import integrator_TDforce 

class TDforceLocal(integrator_TDforce):
    'The (local) Velocity Verlet Integrator.'
    def __init__(self, system, center=[]):
        if not (pmi._PMIComm and pmi._PMIComm.isActive()) or pmi._MPIcomm.rank in pmi._PMIComm.getMPIcpugroup():
            cxxinit(self, integrator_TDforce, system)
            
            # set center of TD force
            if (center != []):
                self.cxxclass.setCenter(self, center[0], center[1], center[2])

    def addForce(self, itype, filename, type):
            """
            Each processor takes the broadcasted interpolation type,
            filename and particle type
            """
            if pmi.workerIsActive():
                self.cxxclass.addForce(self, itype, filename, type)

if pmi.isController :
    class TDforce(object):
        __metaclass__ = pmi.Proxy
        pmiproxydefs = dict(
            cls =  'espresso.integrator.TDforceLocal',
            pmiproperty = [ 'itype', 'filename'],
            pmicall = ['addForce']
            )