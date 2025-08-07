import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianNN_Heteroscedastic, BayesianLinearRegression

from scripts.online_test import test
from scripts.AR_parameterisation import ParameterisationAR1
from utils.concat_files import concat_files

# Set up parameters for simulation
params ={
    'F': 20,
    'c': 10,
    'b': 10,
    'h': 1,
    'J': 32,
    'K': 8,
    'dt': 0.001,
    'dt_f': 0.005,
}
test_params = { 'fname':'run00_X_dtf.npy',
                'runtype': None,
                'save_model_path':'',
                'save_prefix':'',
                'n_ens': 50,
                'N_init': 1,
                'save_step': 1,
                'run_type': 'epistemic',
                'save_prefix': 'epistemic_fix_run00_',
                'T':1000 ,
                'F':20                  }

# Model name
model_name =  f"BayesianNN_16_16_N100" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
test_params['save_model_path'] = model_path

# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]

# Lag-1 Autocorrelation of long timeseries is 0.984865 
phi = 0.984865 
sigma = pyro.get_param_store()['sigma'].detach()


for i in range(10):
    test_params['fname'] = f'run{i:02d}_X_dtf.npy'

    # Run Epistemic with fixed parameters - will sample guide parameters before each ensemble member
    parameterisation_AR1 = ParameterisationAR1(pyro_model, guide, sigma = 0., phi=0.)
    param_sample = parameterisation_AR1.sample_guide_params
    param_func = parameterisation_AR1.keep_epistemic_fixed
    test_params['runtype'] = 'epistemic'
    test_params['save_prefix'] = f'epistemic_fix_run{i:02d}_' 
    test(params, test_params, param_func, param_sample=param_sample)

    # Run Both with fixed parameters - epistemic fixed and aleatoric sampled using AR1
    parameterisation_AR1 = ParameterisationAR1(pyro_model, guide, sigma = sigma, phi=phi)
    param_sample=parameterisation_AR1.sample_guide_params
    param_func = parameterisation_AR1.keep_epistemic_fixed
    test_params['runtype'] = 'both'
    test_params['save_prefix'] = f'both_fix_AR1_run{i:02d}_' 
    test(params, test_params, param_func, param_sample=param_sample)

    # Run Aleatoric with AR1 - need to set up ParameterisationAR1 class with sigma from guide
    parameterisation_AR1 = ParameterisationAR1(pyro_model, guide, sigma = sigma, phi=phi)
    param_func = parameterisation_AR1.aleatoric_only
    test_params['runtype'] = 'aleatoric'
    test_params['save_prefix'] = f'aleatoric_AR1_run{i:02d}_' 
    test(params, test_params, param_func)


