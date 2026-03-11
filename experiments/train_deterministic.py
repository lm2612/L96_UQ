import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import NN, LinearRegression
from scripts.train import train

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
training_params = {'N_train': 100, 
                    'batch_size':128,
                    'N_timesteps':1,
                    'predict_sigma':False,
                    'save_prefix':''}
N_train = training_params['N_train']
seeds = range(100, 150)
for seed in seeds:
    training_params['save_prefix'] = f"seed{seed}"
    np.random.seed(seed)

    model_name =  f"NN_N{N_train}"      # Choose LinearRegression or NN 
    model = NN(1, 1, [16, 16])  
    total_params = sum(p.numel() for p in model.parameters())
    print("TOTAL PARAMS: ", total_params)
    train(params, training_params, model_name, model)