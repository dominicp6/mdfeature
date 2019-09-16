#Install with a clean conda environment

The folowwing are a complete set of instructions in order to install mdfeature without any issues in a new conda environment listing all required dependencies etc. 

1. Create a new conda environment   
```
conda env create -f environment.yml -n mdfeaturen
```
`source activate mdfreature`
2. PyDiffmap: The pip install is incompatible with scipy v 1.+, therefore we'll install from source:   
 `git clone https://github.com/DiffusionMapsAcademics/pyDiffMap.git`   
 `cd pyDiffMap`   
 `git checkout dev_old_scipy`   
 `python setup.py install`   

3. Install mdfeature:   
```
git clone git@github.com:michellab/mdfeature.git
```   
```
cd mdfeature
```   
```
python setup.py install
```


