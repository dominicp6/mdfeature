import subprocess

simulation_length = '250ns'
step_size = '2fs'
save_frequency = '1ps'
number_of_repeats = 3
methods = ['PCA', 'TICA', 'VAMP']

for repeat in range(number_of_repeats):
    for method in methods:
        open('HILLS', 'w').close()
        open('COLVAR', 'w').close()
        subprocess.call(f"./run_openmm.py alanine.pdb amber -d {simulation_length} -f {save_frequency} -s {step_size} -p -mdm {method}", shell=True)
