import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN

from scripts.online_run import online_run
from scripts.Parameterisation import *

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

test_params = { 'fname':'X_dtf.npy',
                'runtype': None,
                'save_model_path':'',
                'save_prefix':'',
                'n_ens': 50,
                'N_init': 100,
                'save_step': 1,
                'T':10 ,
                'F':20                  }

# Model name
model_name =  f"BayesianNN_16_16_N100_priorNormal(0,1.0)" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
test_params['save_model_path'] = model_path

# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]

# Lag-1 Autocorrelation of long timeseries is 0.984865 
phi = 0.984865 

# Set up Parameterisation for Homoscedastic BNN learned via Variational Inference (VI)
parameterisation = Parameterisation_VI(pyro_model, guide=guide, phi=phi)

# White noise
# Epistemic
param_func = parameterisation.WN_param_epistemic
test_params['runtype'] = 'epistemic'
test_params['save_prefix'] = f'VI_WN_epistemic_' 
online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

# Aleatoric
param_func = parameterisation.WN_param_aleatoric
test_params['runtype'] = 'aleatoric'
test_params['save_prefix'] = f'VI_WN_aleatoric_' 
online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

# Both
param_func = parameterisation.WN_param_both
test_params['runtype'] = 'both'
test_params['save_prefix'] = f'VI_WN_both_' 
online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

# AR1
# Epistemic
param_func = parameterisation.AR1_param_epistemic
test_params['runtype'] = 'epistemic'
test_params['save_prefix'] = f'VI_AR1_epistemic_' 
online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

# Aleatoric
param_func = parameterisation.AR1_param_aleatoric
test_params['runtype'] = 'aleatoric'
test_params['save_prefix'] = f'VI_AR1_aleatoric_' 
online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

# Both
param_func = parameterisation.AR1_param_both
test_params['runtype'] = 'both'
test_params['save_prefix'] = f'VI_AR1_both_' 
online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

# Epistemic fixed
param_func = parameterisation.fixed_param_epistemic
test_params['runtype'] = 'epistemic'
test_params['save_prefix'] = f'VI_fixed_epistemic_' 
online_test(params, test_params, param_func, param_sample)

# Both, with epistemic fixed
param_func = parameterisation.fixed_param_both
test_params['runtype'] = 'both'
test_params['save_prefix'] = f'VI_fixed_both_' 
online_test(params, test_params, param_func, param_sample)
