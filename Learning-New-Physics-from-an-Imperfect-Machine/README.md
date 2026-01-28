# Learning New Physics from an Imperfect Machine
## Learn how to propagate systematic uncertainties in the Neyman-Pearson test statistic using machine learning

The exercise studies the 1D toy model proposed in *Learning new physics from an imperfect machine* [(Eur. Phys. J. C 82, 275 (2022)](https://doi.org/10.1140/epjc/s10052-022-10226-y).

The reference (e.g. anomaly-free) data distribution is an exponential distribution with uncertainty affecting the scale and normalization. Uncertainties are modeled by means of two nuisance parameters ($\nu_N,\,\nu_S$):
$$n(x|{\rm R}_{\nu_N,\nu_S})=n({\rm R_0})\exp[-xe^{-\nu_S}-\nu_S+\nu_N]$$

### Jupyter notebooks to learn interactively
This folder contains two notebooks:
- `Parametric_1Dtoy.ipynb` (STEP 1): shows how to learn the parametriation of shape effects on the distribution. This is the first step to run. The parametric model learnt in this notebook is used as input in the second step of the procedure.
- `GOF_1Dtoy.ipynb` (STEP 2): implements the Neyman-Pearson goodness of fit (GOF) test with systematic uncertainties integrated by means of nuisance parameters.

### Python scripts to perform systematic tests
In addition, this folder contains the python scripts to submit multiple toys on a cluster using SLURM:
- `NPLM_1Dtoy.py`: implements the Neyman-Pearson goodness of fit (GOF) test with systematic uncertainties integrated \
by means of nuisance parameters.\
Arguments:
  - `--seed` [int] used to set the toy random seed;
  - `--jsonfile` [path-to-jsonfile] used to pass the path to a json config file containing the experimental settings; this is generated automatically by the submission script (see next bulet) additional details on the json file format is given below. 
- `submit_toys_slurm_FASrc.py`: is used to submit `NPLM_1Dtoy.py` into the SLURM system.\
Arguments:
  - `--pyscript` [str] name of python script to execute;
  - `--toys` [int] number of toys to submit;
  - `--firstseed` [int] if given, the toys are launched with deterministic seed incresing of one unit starting from this value (random seeds are created otherwise);
  - `--local` [0/1] 1 to run locally, 0 to submit with SLURM.

### Example of usage:
```
python submit_toys_slurm_FASrc.py -p NPLM_1Dtoy.py -t 1 -s 1 -l 1
```
#### Example of json config file
```
# configuration dictionary
config_json = {
    "make_plots" : True,
    "verbose"    : True,
    "N_Ref"   : 200000,
    "N_Bkg"   : 2000,
    "N_Sig"   : 0,#10,
    "Sig_loc": 6.4,
    "Sig_std": 0.16,
    "is_tail_excess": False,
    "output_directory": OUTPUT_DIRECTORY,
    "shape_nuisances_id":        shape_nuisances_id_list,
    "shape_nuisances_data":      [nu.nuisances_data[k] for k in shape_nuisances_id_list],
    "shape_nuisances_reference": [nu.nuisances_reference[k] for k in shape_nuisances_id_list],
    "shape_nuisances_sigma":     [nu.nuisances_sigma[k] for k in shape_nuisances_id_list], 
    "shape_models":              [nu.shape_model[k] for k in shape_nuisances_id_list],
    "norm_nuisances_data":       nu.nuisances_data['norm'],
    "norm_nuisances_reference":  nu.nuisances_reference['norm'],
    "norm_nuisances_sigma":      nu.nuisances_sigma['norm'],
    "epochs_tau": 30,
    "patience_tau": 10,
    "epochs_delta": 30,
    "patience_delta": 10,
    "novelty_finder_architecture": [1,4,1],
    "novelty_finder_weight_clipping": 9, 
    "correction": "SHAPE", # choose between "SHAPE", "NORM", ""
}
```
Nuisance parameters are configurated in `utils/NUconfig.py`
