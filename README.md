# Wasserstein Weisfeiler-Lehman graph kernel with Gaussian Process Regression (WWL-GPR)
[![igraph](https://img.shields.io/badge/igraph-0.91-red.svg)](https://igraph.org/) [![ray](https://img.shields.io/badge/Ray-2.0.0-blue.svg)](https://docs.ray.io/en/master/index.html) [![ray](https://img.shields.io/badge/POT-lasted-blue.svg)](https://pythonot.github.io/) 


# Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Overview
This software package provides the implementation of the WWL-GPR model, a data-efficient, physics-inspired machine learning (ML) model for the prediction of binding motifs and associated adsorption enthalpies of complex adsorbates at transition metals (TMs) and their alloys based on a customized Wasserstein Weisfeiler-Lehman graph kernel and Gaussian Process Regression. The task that is solved is to directly predict the relaxed adsorption enthalpies corresponding to a range of plausible initial guesses of the adsorption motif based on graph representation. Thereby, for a given surface/adsorbate combination of interest, both the most stable and all meta-stable adsorption motifs as well as their associated adsorption enthalpies can be predicted. Apart from a graph representation of the intial geometry, the model uses input features in the form of node attributes, which represent physically motivated properties, e.g. d-band moments (surfaces), HOMO/LUMO energy levels (adsorbate molecules) and features of the local geometry, all derived from either the clean surfaces or the adsorbates in the gas phase. Optimization of the hyperparameters in the model is done with Bayesian optimization implemented with scikit-optimize.

A case study predicting adsorption enthalpies of complex adsorbates involved in ethanol synthesis is provided.

Please refer to our manuscript for further details (link to be inserted upon publication).
## System Requirements
The `WWL-GPR` package requires only a standard computer with enough RAM to support the training and prediction of the ML model through the required `conda`  environment (see below). To the benefit of computational scientists likely to accelerate the ML process or interact with other computationally intensive codes on High-Performance Computing (HPC) facility, this package requires a standard `SLURM` Workload Manager.

All software requirements associated with their version are specified in self-contained [env.yml](https://github.com/Wenbintum/WWL-GPR/blob/main/env.yml)



## Installation
The easiest way to install the prerequisites is via [conda](https://conda.io/docs/index.html). All the dependencies are given in `env.yml`.

Firstly, download or clone this repository via:
```bash
git clone https://github.com/Wenbintum/WWL-GPR.git
```

Ensure you have installed conda, step into the WWL-GPR directory, and then run the following command to create a new environment named wwl-gpr.
```bash
conda env create -f env.yml
```
Activate the conda environment with:
```bash
conda activate wwl-gpr
```
Install the package with:
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
We provide three machine learning tasks as showcases in compliance with our manuscript, that are: 
- a) 5-fold Cross-validation applied to in-domain prediction for the complex adsorbates database (termed as "CV5") 
- b) 5-fold Cross-validation applied to in-domain prediction for the simple adsorbates database (termed as "CV5_simpleads")
- c) Extrapolation to out-of-domain samples, an alloy (CuCo) and a new metal (Pt), when only training on the complex adsorbates database containing the elemental metals Cu, Co, Rh and Pt as well as additionally the atomic species (H, O, and C) calculated at Pt (termed as "Extrapolation")  

All tasks can be viewed by running:
```bash
  python main.py -h
```

#### Run task on HPC facility:
By coupling python-interface SLURM scripts and self-contained ML tasks, now you can run these three tasks on High Performance Computing (HPC) facility. For instance, running task a) by given computational resources of 40 CPUs and 3 hours with the title "test".
```bash
  python launch.py --num-cpus 40 -t 03:00:00  --exp-name test --command "python -u main.py --task CV5 --uuid \$redis_password"
```
&emsp;&ensp; Run extrapolation task via:
```bash
  python launch.py --num-cpus 40 -t 03:00:00  --exp-name test --command "python -u main.py --task Extrapolation --uuid \$redis_password"
```
#### Run task on local computer:
We also provide an example for running 5-fold cross-validation within the complex adsorbates database on a local desktop or laptop with fixed hyperparameters (FHP). In this case, the ML learning task will be run on 8 CPUs as given in [input.yml](https://github.com/Wenbintum/WWL-GPR/blob/main/database/complexads_interpolation/input.yml).
- Run task on local desktop or laptop:
```bash
  python main.py --task CV5_FHP
```
We use Bayesian optimization to optimize hyperparameters. You may want to change the settings via [this function](https://github.com/Wenbintum/WWL-GPR/blob/8c52f1f9462215f29ed51517077ea01c077c2d50/wwlgpr/WWL_GPR.py#L302)


#### Expected outpout and run time:

The output file consists of ground truth and ML predicted values, which is located in the "Results" directory for further analysis, and resulting Root Mean Square Error (RMSE) is printed. The run time of CV5_FHP on local computer with 8 CPUs is around 7 minutes, for which the RMSE is of 0.18 eV.


## Authors
This software was primarily written by Wenbin Xu who was advised by Prof. Mie Andersen.

## Acknowledgements
The simple adsorbates database is taken from our previous papers [Deimel et al., ACS Catal. 2020, 10, 22, 13729–13736](https://pubs.acs.org/doi/abs/10.1021/acscatal.0c04045) and [Andersen et al., ACS Catal. 2019, 9, 4, 2752–2759](https://pubs.acs.org/doi/10.1021/acscatal.8b04478).
The complex adsorbates database is constructed via [AIIDA](https://aiida.readthedocs.io/projects/aiida-core/en/latest/) and [CatKit](https://catkit.readthedocs.io/en/latest/)
The WWL-GPR implementation is based on the [Wasserstein Weisfeiler-Lehman Graph Kernels](https://arxiv.org/abs/1906.01277)

## License
WWL-GPR is released under the MIT License.
