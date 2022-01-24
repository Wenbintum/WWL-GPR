# Wasserstein Weisfeiler-Lehman Graph Gaussian Process Regression (WWL-GPR)

This software package provides implementations of Physic-inspired Wasserstein Weisfeiler-Lehman Graph model that takes arbitrary initial guess of structures to predict material properties with a particular interest in handling complex adsorbates. 

This model integrates a customized graph kernel with Gaussian Process Regression. The search of hyperparameters is done with Bayesian optimization implemented with scikit-optimize.

A case study predicting adsorption enthalpies of complex adsorbates involved in Ethanol synthesis is exemplified, where the input features of the Machine learning model are only derived from clean surface and isolated molecule.


## Installation
We made the easiest way to install prerequisites via [conda](https://conda.io/docs/index.html). All the dependencies are given in `env.yml`.

Firstly, download or clone this repository via:
```bash
git clone https://github.com/Wenbintum/WWL-GPR.git
```

Ensure you have installed conda, and then run the following commands to create a new environment named wwl-gpr.
```bash
conda env create -f env.yml
```
Activate the conda environment with:
```bash
conda activate wwl-gpr
```
Step into WWL-GPR directory and install this package with:
```bash
pip install -e .
```

## Usage

This package allows parallel computing at a local computer or a supercomputing facility. We implement this functionality via [Ray](https://docs.ray.io/en/master/index.html), a simple and universal API for building distributed applications. Ray, originally developed by the computer science community, has many benefits, not least providing simple primitives for building and running distributed applications. Readers are referred to the webpage of [Ray](https://docs.ray.io/en/master/index.html) for more details.


#### Python-interface [SLURM](https://slurm.schedmd.com/documentation.html) scripts: 
We provide a helper utility to auto-generate SLURM scripts and launch.  There are some options you can use to submit your job in the SLURM system by running:
```bash
  python launch.py -h
```


#### Available machine learning tasks:
We provide three machine learning tasks as showcases in compliance with our paper, that are: 
- a) 5-fold Cross-validation applied in-domain prediction for complex adsorbates database (termed as "CV5") 
- b) 5-fold Cross-validation applied in-domain prediction for simple adsorbates database (termed as "CV5_simpleads")
- c) Extrapolation to out-of-domain samples, alloy and new metal Pt, when only training on pure metal database and additional including atomic-species on Pt (termed as "Extrapolation")  

All tasks can be viewed by running:
```bash
  python main.py -h
```

#### Run task on HPC facility:
By coupling python-interface SLURM scripts and self-contained ML tasks, now you can run these three tasks on High Performance Computing (HPC) facility. For instance, running task a) by given computational resource of 40 CPUs and 3 hours title with "test".
```bash
  python launch.py --num-cpus 40 -t 03:00:00  --exp-name test --command "python -u main.py --task CV5 --uuid \$redis_password"
```
&emsp;&ensp; Run extrapolation task via:
```bash
  python launch.py --num-cpus 40 -t 03:00:00  --exp-name test --command "python -u main.py --task Extrapolation --uuid \$redis_password"
```
#### Run task on local computer:
We also provide a showcase for running 5-fold Cross-validation of in-domain prediction for complex adsorbates database on local desktop or laptop with fixed hyperparameters (FHP). In this case, the ML learning will be run on 8 CPUS as given in [input.yml](https://github.com/Wenbintum/WWL-GPR/blob/main/database/complexads_interpolation/input.yml).
- Run task on local desktop or laptop:
```bash
  python main.py --task CV5_FHP
```

The output file is located in "Results" directory for further analysis. We use Bayesian optimization to optimize hyperparameters, for which you may want to change its setting via [this function](https://github.com/Wenbintum/WWL-GPR/blob/8c52f1f9462215f29ed51517077ea01c077c2d50/wwlgpr/WWL_GPR.py#L302)

## Authors
This software was primarily written by Wenbin Xu who was advised by Prof. Mie Andersen.

## Acknowledgements
The simple adsorbates database is taken from our previous papers [Deimel et al., ACS Catal. 2020, 10, 22, 13729–13736](https://pubs.acs.org/doi/10.1021/acscatal.8b04478) and [Andersen et al., ACS Catal. 2019, 9, 4, 2752–2759](https://pubs.acs.org/doi/10.1021/acscatal.8b04478).
The complex adsorbates database is constructed via [AIIDA](https://aiida.readthedocs.io/projects/aiida-core/en/latest/) and [CatKit](https://catkit.readthedocs.io/en/latest/)
The WWL-GPR implementation is based on the [Wasserstein Weisfeiler-Lehman Graph Kernels](https://arxiv.org/abs/1906.01277)

## License
PWWG-GPR is released under the MIT License.
