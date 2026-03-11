import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import NN, LinearRegression
from scripts.train import train
from scripts.offline_test import offline_errs

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


# Hyper parameter sweep over lots of different NN architectures
hidden_nodes = [ [8], [16], [32], [64], [8, 8], [16, 16], [32, 32], [64, 64], 
    [8, 8, 8],  [16, 16, 16], [32, 32, 32] ]

seeds = range(100, 101) 

model_names = []
for hidden_node in hidden_nodes:
    print(hidden_node)
    hidden_node_str = f""
    for h in hidden_node:
        hidden_node_str = f"{hidden_node_str}_{h}"
    model_name =  f"NN{hidden_node_str}_N{N_train}" 
    model_names.append(model_name)
    print(model_name)

    for seed in seeds:
        np.random.seed(seed)
        model = NN(1, 1, hidden_node)  
        total_params = sum(p.numel() for p in model.parameters())
        print("TOTAL PARAMS: ", total_params)
        train(params, training_params, model_name, model)

print(offline_errs(params, model_names))

# Loop over activations for the two layer 16,16 model
activations = ["ReLU", "LeakyReLU", "Sigmoid", "Tanh"]
hidden_node = [16, 16]
hidden_node_str = f""
for h in hidden_node:
    hidden_node_str = f"{hidden_node_str}_{h}"

save_prefixs = []
model_names = []

for activation in activations:
    model_name =  f"NN{hidden_node_str}_N{N_train}" 
    model_names.append(model_name)
    save_prefix = f'{activation}_'
    training_params['save_prefix']=save_prefix
    save_prefixs.append(save_prefix)
    np.random.seed(seed)
    model = NN(1, 1, hidden_node, activation=activation)  
    total_params = sum(p.numel() for p in model.parameters())
    print("TOTAL PARAMS: ", total_params)
    train(params, training_params, model_name, model)

print(offline_errs(params, model_names, save_prefixs))
