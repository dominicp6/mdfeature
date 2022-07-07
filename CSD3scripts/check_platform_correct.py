from openmm import LangevinIntegrator, Platform, testInstallation
from openmm.unit import *
from openmm.app import ForceField, Simulation, DCDReporter, PDBFile, CutoffNonPeriodic, HBonds
import mdtraj as md

n = Platform.getNumPlatforms()
print('There are {} platforms.'.format(n))

for i in range(n):
    p = Platform.getPlatform(i)
    print('    {}'.format(p.getName()))

def run_simulation(pdb_name, save_name, iterations, reportInterval, temperature,
                   force_fields=['amber14-all.xml', 'amber14/spce.xml'],platform='CUDA'):
    pdb = PDBFile(pdb_name)

    forcefield = ForceField(*force_fields)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)

    integrator = LangevinIntegrator(temperature * kelvin, friction_coefficient, stepSize)
    simulation = Simulation(pdb.topology, system, integrator, platform=Platform.getPlatformByName(platform))
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature * kelvin)

    mdinit = md.load_pdb(pdb_name)

    mdinit.save_dcd(save_name)
    simulation.reporters.append(DCDReporter(save_name, reportInterval=reportInterval, append=True))

    for i in range(iterations):
        simulation.step(reportInterval)
        state = simulation.context.getState(getEnergy=True, enforcePeriodicBox=False)
        positions = simulation.context.getState(getPositions=True).getPositions()

    return simulation

testInstallation.run_tests()

"""
Analine Dipeptide CSD3 Simulation

timestep: 0.002ps
configuration saved every 0.1ps
simulation time 20000 ps (20 ns)
"""

pdb_name = 'alanine.pdb'
save_name = 'CSD3_alanine_dipeptide_traj.dcd'

stepSize = 0.002 * picoseconds
iterations = 500 #000 * 50
reportInterval = 50

friction_coefficient = 1.0 / picosecond
temperature = 300
R = 0.0083144621  # Universal Gas Constant kJ/K/mol
beta = 1.0 / (temperature * R)  # units (kJ/mol)**(-1)
import time

time0 = time.time()
run_simulation(pdb_name=pdb_name,
               save_name=save_name,
               reportInterval=reportInterval,
               iterations=iterations,
               temperature=temperature,
               platform='CPU')
time1 = time.time()
run_simulation(pdb_name=pdb_name,
               save_name=save_name,
               reportInterval=reportInterval,
               iterations=iterations,
               temperature=temperature,
               platform='CUDA')
time2 = time.time()

print(f"CPU time: {time1-time0}")
print(f"GPU time: {time2-time1}")


