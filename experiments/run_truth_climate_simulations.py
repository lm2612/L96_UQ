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

for F_test in [20, 16, 24]:
    test_params = { 'save_path':f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/",
                    'load_prefix': '',
                    'save_prefix': f'climate_F{F_test}_run00_', 
                    'save_Y': False,
                    'save_ICs': True,
                    'T':1000 ,
                    'F':F_test                  }


    seed = 123
    np.random.seed(seed)
    generate_truth(params, test_params)

    for i in range(0, 10):
        test_params['load_prefix'] = f'run{i:02d}_'
        test_params['save_prefix'] = f'climate_F{test_params["F"]}_run{i+1:02d}_'
        print(f"Loading from {test_params['load_prefix']}, saving to {test_params['save_prefix']}")

        generate_truth(params, test_params)
