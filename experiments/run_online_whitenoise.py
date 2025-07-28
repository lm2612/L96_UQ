import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression


from scripts.online_test import test

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
                'N_init': 20,
                'save_step': 1,
                'T':10 ,
                'F':20                  }

# Model name
model_name =  f"BayesianNN_16_N50" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
test_params['save_model_path'] = model_path

# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]
#predictive = Predictive(pyro_model, guide=guide, num_samples=1, return_sites=("_RETURN", "obs"))

# Run Epistemic with white noise
def param_func(x):
    fixed_param_NN = pyro_model.get_fixed_param_NN(guide())
    fixed_param_NN.eval()
    with torch.no_grad():
        out = fixed_param_NN(x.unsqueeze(-1))
    return out.squeeze()
test_params['runtype'] = 'epistemic'
test_params['save_prefix'] = 'epistemic_' 
test(params, test_params, param_func)

# Run Aleatoric with white noise
fixed_param_NN = pyro_model.get_fixed_param_NN(guide.median())
fixed_param_NN.eval()
def param_func(x):
    with torch.no_grad():
        mean = fixed_param_NN(x.unsqueeze(-1))
        out = pyro_model.sample_obs(mean)
    return out.squeeze()

test_params['runtype'] = 'aleatoric'
test_params['save_prefix'] = 'aleatoric_' 
test(params, test_params, param_func)

# Run both types of uncertainty 
def param_func(x):
    fixed_param_NN = pyro_model.get_fixed_param_NN(guide())
    fixed_param_NN.eval()
    with torch.no_grad():
        mean = fixed_param_NN(x.unsqueeze(-1))
        out = pyro_model.sample_obs(mean)
    return out.squeeze()

test_params['runtype'] = 'both'
test_params['save_prefix'] = 'both_' 
#test(params, test_params, param_func)

# Deterministic - no uncertainty
fixed_param_NN = pyro_model.get_fixed_param_NN(guide.median())
def param_func(x):
    with torch.no_grad():
        mean = fixed_param_NN(x.unsqueeze(-1))
    return mean.squeeze()
test_params['runtype'] = 'deterministic'
test_params['save_prefix'] = 'deterministic_' 
test_params['n_ens'] = 1
#test(params, test_params, param_func)
