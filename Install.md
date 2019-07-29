#Install with a clean conda environment


1. Create a new conda environment   
```
conda create -n mdfeature
```
2. Activate the environment   
```
source activate mdfeature
```
3. Install all required packages:   
```
conda install jupyter scipy scikit-learn seaborn sphinx pyemma mdtraj 
```   
```
consta intall -c omnia openmm openmmtools
```
4. PyDiffmap: The pip install is incompatible with scipy v 1.+, therefore we'll install from source:    
```
git clone https://github.com/DiffusionMapsAcademics/pyDiffMap.git      
git checkout dev_old_scipy   
python setup.py install   
```
5. Install mdfeature:   
```
git clone
```   
```
cd mdfeature
```   
```
python setup.py install
```


