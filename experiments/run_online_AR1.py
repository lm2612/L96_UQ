import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianNN_Heteroscedastic, BayesianLinearRegression

from scripts.online_test import test
from scripts.AR_parameterisation import ParameterisationAR1

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
                'N_init': 1,
                'save_step': 1,
                'T':10 ,
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
parameterisation_AR1 = ParameterisationAR1(pyro_model, guide, sigma = 0., phi=phi)

# Run Epistemic treated as AR1
param_func = parameterisation_AR1.epistemic_AR1
test_params['runtype'] = 'epistemic'
test_params['save_prefix'] = 'epistemic_AR1_' 
#test(params, test_params, param_func)

# Run Aleatoric with AR1 - need to set up ParameterisationAR1 class with sigma from guide
sigma = pyro.get_param_store()['sigma'].detach()
parameterisation_AR1 = ParameterisationAR1(pyro_model, guide, sigma = sigma, phi=phi)
param_func = parameterisation_AR1.aleatoric_only

test_params['runtype'] = 'aleatoric'
test_params['save_prefix'] = 'aleatoric_AR1_' 
test(params, test_params, param_func)

parameterisation_AR1 = ParameterisationAR1(pyro_model, guide, sigma = sigma, phi=phi, 
    aleatoric=False, epistemic=True, N=10)
param_func = parameterisation_AR1.AR1_param
test_params['runtype'] = 'epistemic'
test_params['save_prefix'] = 'new_epistemic_AR1_' 
test(params, test_params, param_func)

parameterisation_AR1 = ParameterisationAR1(pyro_model, guide, sigma = sigma, phi=phi, 
    aleatoric=True, epistemic=True, N=10)
param_func = parameterisation_AR1.AR1_param
test_params['runtype'] = 'both'
test_params['save_prefix'] = 'new_both_AR1_' 
test(params, test_params, param_func)
