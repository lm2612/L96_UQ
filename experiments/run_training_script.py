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
training_params = {'N_train': 50, 
                    'batch_size':128,
                    'N_timesteps':1}
N_train = training_params['N_train']
seeds = range(100, 101) #, 150)
for seed in seeds:
    model_name =  f"LinearRegression_N{N_train}"      # Choose LinearRegression or NN 
    np.random.seed(seed)
    model = LinearRegression(1, 1) #, dropout_rate=0.5)
    total_params = sum(p.numel() for p in model.parameters())
    print("TOTAL PARAMS: ", total_params)
    train(params, training_params, model_name, model)