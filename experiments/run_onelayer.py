import os 
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
                'n_ens': 1,
                'N_init': 2000,
                'save_step': 1,
                'T':2*0.005 ,
                'F':20                  }

# Model name
model_name =  f"OneLayer" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
print(model_path)
test_params['save_model_path'] = model_path

# Return zeros
def param_func(x):
    return torch.zeros_like(x)
test_params['runtype'] = 'zero'
test_params['save_prefix'] = 'offline_zero_' 
#test(params, test_params, param_func)

test_params['N_init'] = 160
test_params['T'] = 0.125
test_params['save_prefix'] = 'zero_' 
test(params, test_params, param_func)