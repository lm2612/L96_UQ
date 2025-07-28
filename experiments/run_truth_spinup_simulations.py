import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression

from scripts.generate_test_data import generate_truth

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

test_params = { 'save_path':f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/",
                'load_prefix': 'rand00_',
                'save_prefix': 'from_rand00_', 
                'save_Y': False,
                'save_ICs': False,
                'T':10 ,
                'F':20                  }


seed = 123
np.random.seed(seed)
# Generate random initial conditions 
X_min = -4
X_max = 10
rng = np.random.default_rng(seed)

X0 = rng.uniform(X_min, X_max, params['K'])
Y0 = rng.uniform(X_min, X_max, params['K']*params['J'])
# Save
np.save(f"{test_params['save_path']}/{test_params['load_prefix']}X_init.npy", X0) 
np.save(f"{test_params['save_path']}/{test_params['load_prefix']}Y_init.npy", Y0) 

generate_truth(params, test_params)

