checklist:
1. simple_adsorbate_db.zip
include 1422 optimized structures of 8 simple adsorbates on various surfaces and their adsorption energies 

2. complex_adsorbate_db.zip
include 2117 optimized sturctures of 41 small to complex adosrbates on pure metal Cu, Co, Pd, Rh, Pt, Ru and alloy CuCo, PdRh surfaces and their adsorption energies.
we use 1679 (Cu, Co, Pd, Rh) out of 2117 samples performing interpolation task, remaining samples (Pt, CuCo, Ru, PdRh) for the purpose of extrapolation

3. we provide some pre-defined ML tasks for the purpose of tutorial and data reproducing, please follow the instruction on the main page and
you are referred to our paper for detailed explanation of the tasks and their setting.

4. we provide a tutorial about how to construct SOAP descriptor to represent local geometry (which is one of the important ingredients of WWL-GPR model)

5. we also provide a code snippet to generate work function and d band moments. These are basically postprocessing step of the output files. Given different DFT code and version, they can differ a lot. People should test and adapt it into their case study.  

6. we provide a tutorial about the implementation of XGBoost method and its hyperparameter search.
