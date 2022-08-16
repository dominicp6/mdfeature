# from openmm import *
# from openmm.app import *
# from openmm.unit import *
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.image as image
# from pyemma import msm
# from scipy.interpolate import griddata
# from matplotlib.pyplot import cm
# import mdtraj as md
# from ipywidgets import IntProgress
# from IPython.display import display
# import time
# import numpy as np
#
# import mdfeature.features as features
# from pydiffmap_weights.LSD_diffusion_map import LSDMap
#
# import matplotlib.pyplot as plt
# import matplotlib as mpl
#
# if __name__ == "__main__":
#     mpl.rcParams['lines.linewidth'] = 2
#     font = {'family' : 'sans-serif',
#             'size'   : 14.0}
#     mpl.rc('font', **font)
#
#     mpl.rcParams['xtick.labelsize'] = 16
#     mpl.rcParams['ytick.labelsize'] =  16
#     mpl.rcParams['font.size'] =  15
#     mpl.rcParams['figure.autolayout'] =  True
#     mpl.rcParams['figure.figsize'] =  7.2,4.45
#     mpl.rcParams['axes.titlesize'] =  16
#     mpl.rcParams['axes.labelsize'] =  17
#     mpl.rcParams['lines.linewidth'] =  2
#     mpl.rcParams['lines.markersize'] =  6
#     mpl.rcParams['legend.fontsize'] =  13
#
#
#     # Bunch of useful simulation parameters
#     steps = 100
#     iterations = 100
#     beta = 1.0/(300.0*0.0083144621)
#
#     pdb_name = 'alanine.pdb'
#     pdb = PDBFile(pdb_name)
#     saving_file = 'trajectory.dcd'
#
#     forcefield = ForceField('amber99sbildn.xml') #amber14-all.xml', 'amber14/spce.xml')
#     system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)
#
#     integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.002*picoseconds)
#     simulation = Simulation(pdb.topology, system, integrator, platform=Platform.getPlatformByName('CPU'))
#     simulation.context.setPositions(pdb.positions)
#     simulation.context.setVelocitiesToTemperature(300*kelvin)
#
#
#     # check the loaded topology
#
#     print(pdb.topology)
#     topology = md.load(pdb_name).topology
#     print(topology)
#
#     table, bonds = topology.to_dataframe()
#     #print(table)
#
#
#     import mdfeature.features as features
#
#     phi = [4, 6, 8 ,14] #dihedral coordinates # [4, 6, 8 ,14]#[1, 6, 8, 14]
#     psi = [6, 8, 14, 16]
#
#     phi_name = features.get_name_torsion(phi, pdb_file=pdb_name, table=table)
#     psi_name = features.get_name_torsion(psi, pdb_file=pdb_name, table=table)
#
#     print('Long name')
#     print(phi_name)
#     print(psi_name)
#
#     phi_name_short = features.get_name_torsion(phi, pdb_file=pdb_name, table=table, format='short')
#     psi_name_short = features.get_name_torsion(psi, pdb_file=pdb_name, table=table, format='short')
#
#     print('Short name')
#     print(phi_name_short)
#     print(psi_name_short)
#
#     # if False, dont run
#     run = True
#
#     mdinit = md.load_pdb(pdb_name)
#
#     if run:
#
#         mdinit.save_dcd(saving_file)
#         simulation.reporters.append(DCDReporter(saving_file, steps, append=True))
#         simulation.reporters.append(PDBReporter('output.pdb', steps))
#
#         max_count = iterations
#         bar = IntProgress(min=0, max=max_count)  # instantiate the bar
#         display(bar)  # display the bar
#
#         for i in range(iterations):
#             bar.value += 1
#             simulation.step(steps)
#
#             state = simulation.context.getState(getEnergy=True, enforcePeriodicBox=False)
#             positions = simulation.context.getState(getPositions=True).getPositions()
#
#
#     traj_std_tmp = md.load_dcd('test_traj.dcd', mdinit.topology)
#
#     skip_first = 1000
#     traj_orig = traj_std_tmp[skip_first:]
#     print(traj_orig)
#     traj_orig = traj_orig.superpose(traj_orig[0])
#
#     import mdfeature.diffusionmap as diffusionmap
#
#
#     #mydmap, traj = diffusionmap.compute_diffusionmaps(traj_orig, nrpoints=2000, epsilon=1.0, type='LSDmap')
#     mydmap = LSDMap(epsilon=1.0, alpha=0.5, k=64, metric='euclidean')
#     mydmap.fit(pdb_trajectory='output.pdb')
#
#     evec = np.asarray(mydmap.evecs[:,0])
#
#     phi = [4, 6, 8 ,14] #dihedral coordinates
#     psi = [6, 8, 14, 16]
#     zeta = [1, 4, 6, 8]
#     theta = [8, 14, 16, 18]
#
#     default_torsions = [phi, psi, zeta, theta]
#     all_combinations = features.create_torsions_list(atoms=traj.xyz.shape[1], size=100, append_to=default_torsions, print_list=True)
#
#     dimension = 2
#
#     list_of_functions =['compute_cos_torsion_mdraj' for _ in range(len(all_combinations))]
#     print(traj)
#     print(mydmap)
#     print(list_of_functions)
#     print(all_combinations)
#     correlations = features.compute_all_correlations(traj, mydmap, dimension, list_of_functions, nevery=10, list_of_params=all_combinations)
#     print(correlations)


from openmm import *
from openmm.app import *
from openmm.unit import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as image
from pyemma import msm
from scipy.interpolate import griddata
from matplotlib.pyplot import cm
import mdtraj as md
from ipywidgets import IntProgress
from IPython.display import display
import time
import numpy as np

import mdfeature.diffusionmap as diffusionmap

from scipy.optimize import dual_annealing

pdb_name = 'alanine.pdb'
save_name = 'trajectory.dcd'

steps = 100
iterations = 100

temperature = 300
beta = 1.0 / (temperature * 0.0083144621)


def run_simulation(pdb_name, save_name, iterations, steps, temperature,
                   force_fields=['amber14-all.xml', 'amber14/spce.xml']):
    pdb = PDBFile(pdb_name)

    forcefield = ForceField(*force_fields)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=CutoffNonPeriodic, constraints=HBonds)

    integrator = LangevinIntegrator(temperature * kelvin, 1.0 / picosecond, 0.002 * picoseconds)
    simulation = Simulation(pdb.topology, system, integrator, platform=Platform.getPlatformByName('CPU'))
    simulation.context.setPositions(pdb.positions)
    simulation.context.setVelocitiesToTemperature(temperature * kelvin)

    mdinit = md.load_pdb(pdb_name)

    mdinit.save_dcd(save_name)
    simulation.reporters.append(DCDReporter(save_name, steps, append=True))

    max_count = iterations
    bar = IntProgress(min=0, max=max_count)  # instantiate the bar
    display(bar)  # display the bar

    for i in range(iterations):
        bar.value += 1
        simulation.step(steps)
        state = simulation.context.getState(getEnergy=True, enforcePeriodicBox=False)
        positions = simulation.context.getState(getPositions=True).getPositions()

    return simulation

mdinit = md.load_pdb(pdb_name)
simulation = run_simulation(pdb_name='alanine.pdb', save_name='trajectory.dcd', iterations=iterations, steps=steps, temperature=temperature)


def open_trajectory_file(traj_file, skip_first=1000, use_test_trajectory=False):
    if use_test_trajectory:
        traj_file = 'miscdata/test_traj.dcd'

    traj_std_tmp = md.load_dcd(traj_file, mdinit.topology)
    traj_orig = traj_std_tmp[skip_first:]
    traj_orig = traj_orig.superpose(traj_orig[0])

    return traj_orig

traj = open_trajectory_file(traj_file=save_name, skip_first=10, use_test_trajectory=True)


weight_params = {}
weight_params['simulation'] = simulation
weight_params['temperature'] = temperature

dmap_obj, traj_final = diffusionmap.compute_diffusionmaps(traj, nrpoints=2000, epsilon=1.0, weights='compute', weight_params=weight_params)

diffusion_coordinate = 0


# TODO: understand theory behind this step
def compute_marginalised_free_energy_from_diffusion_map(diffusion_map, diffusion_coordinate):
    free_energy_counts, coordinate = np.histogram(diffusion_map.dmap[:, diffusion_coordinate], bins=200)
    with numpy.errstate(divide='ignore'):
        free_energy = -np.log(free_energy_counts)

    return free_energy, coordinate[:-1]

free_energy, coordinate = compute_marginalised_free_energy_from_diffusion_map(dmap_obj, diffusion_coordinate)
plt.plot(coordinate, free_energy, 'k')
plt.xlabel('DC 1', fontsize=16)
plt.ylabel('Free Energy', fontsize=16)

from matplotlib.pyplot import figure
diffusion_coordinate_time_series = dmap_obj.dmap[:,diffusion_coordinate]
fig = plt.figure(figsize=(16,2))
plt.plot(diffusion_coordinate_time_series, 'k')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('DC 1', fontsize=16)


def compute_N_and_P(time_series, observation_interval, cells):
    min_coord = min(time_series)
    max_coord = max(time_series)
    cell_sequence = []
    for coord in time_series:
        cell_sequence.append(round((cells - 1) * (coord - min_coord) / (max_coord - min_coord)))

    N = np.zeros((cells, cells))
    state_sequence = [cell for idx, cell in enumerate(cell_sequence) if idx % observation_interval == 0]
    for idx, cell in enumerate(state_sequence[:-1]):
        N[cell, state_sequence[idx + 1]] += 1

    P = np.zeros(cells)
    for state in cell_sequence:
        P[state] += 1

    P = P / np.sum(P)

    return N, P


t = 5  # the subsampling factor ("observation interval")

N, P = compute_N_and_P(time_series=diffusion_coordinate_time_series, observation_interval=t, cells=30)

fig = plt.figure(figsize=(7, 7))
plt.imshow(N)
plt.title(r'$N_{ij}$', fontsize=16)
plt.xlabel('i', fontsize=16)
plt.ylabel('j', fontsize=16)
plt.colorbar()
plt.show()

plt.plot(P)
plt.xlabel('i', fontsize=16)
plt.ylabel('P(i)', fontsize=16)


def negative_log_likelihood(R, *args):
    # R is a 1D array
    N = args[0]
    P = args[1]
    t = args[2]
    number_of_cells = N.shape[0]

    # convert R into a cell x cell matrix
    R = np.reshape(R, (number_of_cells, number_of_cells))
    for p in P:
        if p <=0:
            print(f'Got a prob {p}')
            input()
    P_half = P ** (1 / 2)
    P_minus_half = P ** (-1 / 2)

    # column-wise multiplication by a vector
    intermediate = np.multiply(R, P_half)
    # row-wise multiplication by a vector
    R_tilde = np.multiply(P_minus_half[:, np.newaxis], intermediate)

    Lambda, U = np.linalg.eig(R_tilde)
    e_tLambda = np.diag(np.exp(t * Lambda))
    e_tR_tilde = U @ e_tLambda @ U.transpose()
    e_tR = np.diag(P ** (1 / 2)) @ e_tR_tilde @ np.diag(P ** (-1 / 2))
    ln_e_tR = np.log(e_tR)

    score = - np.sum(np.multiply(N, ln_e_tR))
    print(score)

    return score #- np.sum(np.multiply(N, ln_e_tR))


bounds = [(0,1) for k in range(N.shape[0]*N.shape[1])]
ret = dual_annealing(negative_log_likelihood, bounds=bounds, args=(N,P,t))