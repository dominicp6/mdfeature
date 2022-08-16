import subprocess
import os

simulation_length = '50ns'
step_size = '2fs'
save_frequency = '1ps'
number_of_repeats = 5
methods = ['TICA', 'VAMP', 'DMAP'] #'PCA'

for repeat in range(number_of_repeats):
    for method in methods:
        #open('../HILLS', 'w').close()
        #open('../COLVAR', 'w').close()
        subprocess.call(f"./run_openmm.py alanine.pdb amber -d {simulation_length} -f {save_frequency} -s {step_size} "
                        f"-p -mdm {method} -repeat {repeat+1} -seed {repeat}", shell=True)
