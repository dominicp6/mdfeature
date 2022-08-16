import subprocess
import os

dirs = [x[0] for x in os.walk("/CSD3scripts/outputs")]
print(dirs)

keywords = ['PCA1','PCA2','PCA3','PCA4','PCA5','VAMP1','VAMP2','VAMP3','VAMP4','VAMP5','TICA1','TICA2','TICA3','TICA4','TICA5','DMAP1','DMAP2','DMAP3','DMAP4','DMAP5']

for idx, dir in enumerate(dirs):
    if idx == 0:
        continue
    current_keyword = ''
    with open(f"{dir}/readme.txt", "r") as f:
        read_me_lines = f.readlines()
        for line in read_me_lines:
            if any([keyword in line for keyword in keywords]):
                for keyword in keywords:
                    if keyword in line:
                        print(keyword)
                        subprocess.call(f"mdconvert {dir}/trajectory.dcd -o {dir}/trajectory.xtc", shell=True)
                        subprocess.call(f"plumed driver --mf_xtc {dir}/trajectory.xtc --plumed plumed_reweight_{keyword}.dat --kt 2.479", shell=True)
                        break
