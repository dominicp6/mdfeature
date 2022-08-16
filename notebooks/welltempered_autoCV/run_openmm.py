#!/usr/bin/env python3
import openmm
import openmm.app as app
import openmm.unit as unit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import argparse
import datetime
import os
import sys
from openmmplumed import PlumedForce

##############################################
#   CONSTANTS
##############################################

CHECKPOINT_FN = "checkpoint.chk"
TRAJECTORY_FN = "trajectory.dcd"
STATE_DATA_FN = "state_data.csv"

valid_ffs = ['ani2x', 'ani1ccx', 'amber', "ani2x_mixed", "ani1ccx_mixed"]

# basic quantity string parsing ("1.2ns" -> openmm.Quantity)
unit_labels = {
    "us": unit.microseconds,
    "ns": unit.nanoseconds,
    "ps": unit.picoseconds,
    "fs": unit.femtoseconds
}


def parse_quantity(s):
    try:
        u = s.lstrip('0123456789.')
        v = s[:-len(u)]
        return unit.Quantity(
            float(v),
            unit_labels[u]
        )
    except Exception:
        raise ValueError(f"Invalid quantity: {s}")


##############################################
#   PARSE ARGS
##############################################

parser = argparse.ArgumentParser(description='Production run for an equilibrated peptide.')
parser.add_argument("pdb", help="PDB file describing topology and positions. Should be solvated and equilibrated")
parser.add_argument("ff", help=f"Forcefield/Potential to use: {valid_ffs}")
parser.add_argument("-r", "--resume", help="Resume simulation from an existing production directory")
parser.add_argument("-g", "--gpu", default="", help="Choose CUDA device(s) to target [note - ANI must run on GPU 0]")
parser.add_argument("-d", "--duration", default="1ns", help="Duration of simulation")
parser.add_argument("-f", "--savefreq", default="1ps", help="Interval for all reporters to save data")
parser.add_argument("-s", "--stepsize", default="2fs", help="Integrator step size")
parser.add_argument("-c", "--frictioncoeff", default="1ps",
                    help="Integrator friction coeff [your value]^-1 ie for 0.1fs^-1 put in 0.1fs. The unit but not the value will be converted to its reciprocal.")
parser.add_argument("-np", "--nonperiodic", action=argparse.BooleanOptionalAction,
                    help="Prevent periodic boundary conditions from being applied")
parser.add_argument("-p", "--plumed", action=argparse.BooleanOptionalAction, help="Whether or not to run PLUMED metadynamics.")
parser.add_argument("-mdm", "--metadynamics_method", default='TICA', help="The method to use for metadynamics: either PCA, TICA, VAMP or DMAP.")
parser.add_argument("-repeat", "--repeat_number", default="1", help="The repeat number to use for PLUMED metadynamics integration.")
parser.add_argument("-seed", "--random_number_seed", default="1")
parser.add_argument("-name", "--experiment_name", default=None)
parser.add_argument("-dir", "--directory", default=os.getcwd())

args = parser.parse_args()

pdb = args.pdb
forcefield = args.ff.lower()
resume = args.resume
duration = parse_quantity(args.duration)
savefreq = parse_quantity(args.savefreq)
stepsize = parse_quantity(args.stepsize)
frictioncoeff = parse_quantity(args.frictioncoeff)
plumed = args.plumed

frictioncoeff = frictioncoeff._value / frictioncoeff.unit

total_steps = int(duration / stepsize)
steps_per_save = int(savefreq / stepsize)

if forcefield not in valid_ffs:
    print(f"Invalid forcefield: {forcefield}, must be {valid_ffs}")
    quit()

if resume:
    if not os.path.isdir(resume):
        print(f"Production directory to resume is not a directory: {resume}")
        quit()

    # Check all required files exist in prod directory to resume
    resume_contains = os.listdir(resume)
    resume_requires = (
        CHECKPOINT_FN,
        TRAJECTORY_FN,
        STATE_DATA_FN
    )

    if not all(filename in resume_contains for filename in resume_requires):
        print(f"Production directory to resume must contain files with the following names: {resume_requires}")
        quit()

    # Use existing output directory
    output_dir = resume
else:
    # Make output directory
    pdb_filename = os.path.splitext(os.path.basename(pdb))[0]
    if args.experiment_name is None:
        output_dir = f"production_{pdb_filename}_{forcefield}_{datetime.datetime.now().strftime('%H%M%S_%d%m%y')}"
    else:
        output_dir = f"{args.experiment_name}"
    output_dir = os.path.join(args.directory, output_dir)
    os.makedirs(output_dir)

##############################################
#   CREATE SYSTEM
##############################################

# Load peptide
pdb = app.PDBFile(pdb)
if args.nonperiodic:
    pdb.topology.setPeriodicBoxVectors(None)

peptide_indices = [
    atom.index
    for atom in pdb.topology.atoms()
    if atom.residue.name != "HOH"
]

# for convenience, create "topology.pdb" of the raw peptide, as it is saved in the dcd.
# this is helpful for analysis scripts which rely on it down the line
modeller = app.Modeller(
    pdb.topology,
    pdb.positions
)
modeller.deleteWater()
pdb.writeFile(
    modeller.getTopology(),
    modeller.getPositions(),
    open(os.path.join(output_dir, "topology.pdb"), "w")
)


def makeSystem(ff):
    return ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1 * unit.nanometer,
        # constraints = app.AllBonds,
        # hydrogenMass = 4*unit.amu,
    )


if forcefield == "amber":  # Create AMBER system
    system = makeSystem(app.ForceField(
        'amber14-all.xml',
        'amber14/tip3p.xml'
    ))
else:
    raise ValueError(f'Force field {forcefield} not supported.')

if plumed:
    from pathlib import Path
    script_name = args.metadynamics_method + str(int(args.repeat_number)-1)
    script = Path(f"{args.directory}/{script_name}.txt").read_text()
    system.addForce(PlumedForce(script))
    print(f"Running with PLUMED... ({args.metadynamics_method})")

with open(os.path.join(output_dir, "readme.txt"), "w") as f:
    [print(arg) for arg in args._get_kwargs()]
    [f.write(f"{arg[0]} : {arg[1]}\n") for arg in args._get_kwargs()]
    f.write("\n")
    f.write("PLUMED SCRIPT \n")
    f.write(script)

##############################################
#   INITIALISE SIMULATION
##############################################

print("Initialising production run...")

properties = {'CudaDeviceIndex': args.gpu}

# Create constant temp integrator
integrator = openmm.LangevinMiddleIntegrator(
    300 * unit.kelvin,
    frictioncoeff,
    stepsize
)

# Set random seed
integrator.setRandomNumberSeed(seed=int(args.random_number_seed))

# Create simulation and set initial positions
simulation = app.Simulation(
    pdb.topology,
    system,
    integrator,
    openmm.Platform.getPlatformByName("CUDA"),
    properties
)

simulation.context.setPositions(pdb.positions)
if resume:
    with open(os.path.join(output_dir, CHECKPOINT_FN), "rb") as f:
        simulation.context.loadCheckpoint(f.read())
        print("Loaded checkpoint")

##############################################
#   DATA REPORTERS
##############################################

# Reporter to print info to stdout
simulation.reporters.append(app.StateDataReporter(
    sys.stdout,
    steps_per_save,
    progress=True,  # Info to print. Add anything you want here.
    remainingTime=True,
    speed=True,
    totalSteps=total_steps,
))
# Reporter to log lots of info to csv
simulation.reporters.append(app.StateDataReporter(
    os.path.join(output_dir, STATE_DATA_FN),
    steps_per_save,
    step=True,
    time=True,
    speed=True,
    temperature=True,
    potentialEnergy=True,
    kineticEnergy=True,
    totalEnergy=True,
    append=True if resume else False
))
# Reporter to save trajectory
# Save only a subset of atoms to the trajectory, ignore water
simulation.reporters.append(app.DCDReporter(
    os.path.join(output_dir, TRAJECTORY_FN),
    reportInterval=steps_per_save,
    append=True if resume else False))

# Reporter to save regular checkpoints
simulation.reporters.append(app.CheckpointReporter(
    os.path.join(output_dir, CHECKPOINT_FN),
    steps_per_save
))

##############################################
#   PRODUCTION RUN
##############################################

print("Running production...")
simulation.step(total_steps)
print("Done")

# Save final checkpoint and state
simulation.saveCheckpoint(os.path.join(output_dir, CHECKPOINT_FN))
simulation.saveState(os.path.join(output_dir, 'end_state.xml'))

# Make some graphs
report = pd.read_csv(os.path.join(output_dir, STATE_DATA_FN))
report = report.melt()

with sns.plotting_context('paper'):
    g = sns.FacetGrid(data=report, row='variable', sharey=False)
    g.map(plt.plot, 'value')
    # format the labels with f-strings
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(
            tkr.FuncFormatter(lambda x, p: f'{(x * stepsize).value_in_unit(unit.nanoseconds):.1f}ns'))
    plt.savefig(os.path.join(output_dir, 'graphs.png'), bbox_inches='tight')

# print a trajectory of the aaa dihedrals, counting the flips
# heatmap of phi and psi would be a good first analysis, use mdanalysis
# aiming for https://docs.mdanalysis.org/1.1.0/documentation_pages/analysis/dihedrals.html
# number of events going between minima states
# "timetrace" - a plot of the dihedral over time (aim for 500ns)
# do this first, shows how often you go back and forth. one plot for each phi/psi angle
# four plots - for each set of pairs
# this gives two heatmap plots like in the documentation