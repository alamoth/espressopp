import espressopp
import time
import sys

########################################################################
# 1. specification of the main simulation parameters                   #
########################################################################

# number of particles
Npart              = 100000 #1024
# density of particles
rho                = 0.8442
# length of simulation box
L                  = pow(Npart/rho, 1.0/3.0)
# cubic simulation box of size L
mult               = (2,2,2)
# mult               = (1,1,1)
box                = (mult[0]*L, mult[1]*L, mult[2]*L)
# cutoff of the short range potential
r_cutoff           = 2.5
# VerletList skin size (also used for domain decomposition)
skin               = 0.4
# the temperature of the system
temperature        = 1.0
# time step for the velocity verlet integrator
dt                 = 0.005
# Lennard Jones epsilon during equilibration phase
epsilon            = 1.0
# Lennard Jones sigma during warmup and equilibration
sigma              = 1.0
# number of equilibration loops
equil_nloops       = 1 #10 #1 #20 #10
# number of integration steps performed in each equilibration loop
equil_isteps       = 1000 # 100

# print ESPResSo++ version and compile info
print espressopp.Version().info()
# print simulation parameters (useful to have them in a log file)
print "Npart              = ", Npart
print "rho                = ", rho
print "L                  = ", L
print "box                = ", box 
print "r_cutoff           = ", r_cutoff
print "skin               = ", skin
print "temperature        = ", temperature
print "dt                 = ", dt
print "epsilon            = ", epsilon
print "sigma              = ", sigma
print "equil_nloops       = ", equil_nloops
print "equil_isteps       = ", equil_isteps

########################################################################
# 2. setup of the system, random number geneartor and parallelisation  #
########################################################################

# create the basic system
system             = espressopp.System()
# use the random number generator that is included within the ESPResSo++ package
system.rng         = espressopp.esutil.RNG()
# use orthorhombic periodic boundary conditions 
system.bc          = espressopp.bc.OrthorhombicBC(system.rng, box)
# set the skin size used for verlet lists and cell sizes
system.skin        = skin
# get the number of CPUs to use
NCPUs              = espressopp.MPI.COMM_WORLD.size
# calculate a regular 3D grid according to the number of CPUs available
nodeGrid           = espressopp.tools.decomp.nodeGrid(NCPUs,box,r_cutoff, skin)
# calculate a 3D subgrid to speed up verlet list builds and communication
cellGrid           = espressopp.tools.decomp.cellGrid(box, nodeGrid, r_cutoff, skin)
# create a domain decomposition particle storage with the calculated nodeGrid and cellGrid
system.storage     = espressopp.storage.DomainDecomposition(system, nodeGrid, cellGrid)


print "NCPUs              = ", NCPUs
print "nodeGrid           = ", nodeGrid
print "cellGrid           = ", cellGrid

########################################################################
# 3. setup of the integrator and simulation ensemble                   #
########################################################################

# use a velocity Verlet integration scheme
integrator     = espressopp.integrator.VelocityVerlet(system)
# set the integration step  
integrator.dt  = dt

# use a thermostat if the temperature is set
if (temperature != None):
  # create e Langevin thermostat
  thermostat             = espressopp.integrator.LangevinThermostat(system)
  # set Langevin friction constant
  thermostat.gamma       = 1.0
  # set temperature
  thermostat.temperature = temperature
  # tell the integrator to use this thermostat
  integrator.addExtension(thermostat)

# GPUSupport = espressopp.integrator.GPUTransfer(system)
# integrator.addExtension(GPUSupport)

## steps 2. and 3. could be short-cut by the following expression:
## system, integrator = espressopp.standard_system.Default(box, warmup_cutoff, skin, dt, temperature)

########################################################################
# 4. adding the particles                                              #
########################################################################
# f = open('32768Eq')
# f = open('100000Eq')
# f = open(str(Npart)+'Eq')
f = open('/gpfs/fs2/project/zdvhpc/alamothParticles/'+str(Npart)+'Eq.xyz')
lines = f.readlines()
pos_x = []
pos_y = []
pos_z = []
print "adding ", Npart*mult[0]*mult[1]*mult[2], " particles to the system ..." 
for pid in range(Npart):
  # get a 3D random coordinate within the box
  row = lines[pid].split()
  #pos = system.bc.getRandomPos()
  pos_x.append(float(row[2]))
  pos_y.append(float(row[3]))
  pos_z.append(float(row[4]))
  
  # pos =  espressopp.Real3D(float(row[2]), float(row[3]), float(row[4]))

  # add a particle with particle id pid and coordinate pos to the system
  # coordinates are automatically folded according to periodic boundary conditions
  # the following default values are set for each particle:
  # (type=0, mass=1.0, velocity=(0,0,0), charge=0.0)
  # system.storage.addParticle(pid, pos)

espressopp.tools.replicate_add_particles(system.storage, pos_x, pos_y, pos_z, L, L, L, mult[0], mult[1], mult[2])
# distribute the particles to parallel CPUs 
system.storage.decompose()

########################################################################
# 7. setting up interaction potential for the equilibration            #
########################################################################

# create a new verlet list that uses a cutoff radius = r_cutoff
# the verlet radius is automatically increased by system.skin (see system setup)
verletlist  = espressopp.VerletList(system, r_cutoff)
# define a Lennard-Jones interaction that uses a verlet list 

interaction = espressopp.interaction.VerletListLennardJones(verletlist)

# use a Lennard-Jones potential between 2 particles of type 0 
# the potential is automatically shifted so that U(r=cutoff) = 0.0
# if the potential should not be shifted set shift=0.0
potential = interaction.setPotential(type1=0, type2=0, potential=espressopp.interaction.LennardJones(epsilon=epsilon, sigma=sigma, cutoff=r_cutoff, shift=0.0))

                                  

########################################################################
# 8. running the equilibration loop                                    #
########################################################################

# add the new interaction to the system
system.addInteraction(interaction)
# since the interaction cut-off changed the size of the cells that are used
# to speed up verlet list builds should be adjusted accordingly 
#system.storage.cellAdjust()

# set all integrator timers to zero again (they were increased during warmup)
integrator.resetTimers()
# set integrator time step to zero again
integrator.step = 0

start_time = time.clock()

print "starting equilibration ..."
# print inital status information
espressopp.tools.analyse.info(system, integrator)
#sock = espressopp.tools.vmd.connect(system)
for step in range(equil_nloops):

  # perform equilibration_isteps integration steps
  integrator.run(equil_isteps)
  #espressopp.tools.vmd.imd_positions(system, sock)
  # print status information
  espressopp.tools.analyse.info(system, integrator)
print "equilibration finished"
end_time = time.clock()
# GPUSupport.disconnect()

espressopp.tools.analyse.final_info(system, integrator, verletlist, start_time, end_time)

sys.stdout.write('Eq time = %f\n' % (end_time - start_time))

########################################################################
# 9. writing configuration to file                                     #
########################################################################

# write folded xyz coordinates and particle velocities into a file
# format of xyz file is:
# first line      : number of particles
# second line     : box_Lx, box_Ly, box_Lz
# all other lines : ParticleID  ParticleType  x_pos  y_pos  z_pos  x_vel  y_vel  z_vel 
# filename = "lennard_jones_fluid_C_%f.xyz" % time.clock()
# print "writing final configuration file ..." 
# espressopp.tools.writexyz(filename, system, velocities = True, unfolded = False)

print "finished."
