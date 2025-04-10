import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam

# Get paths
# Define dimensions of system (fixed)
K = 8   
J = 32  

# Define the "true" parameters
h = 1
F = 20  # 8
c = 10
b = 10

# Define time-stepping, random seed
dt = 0.001
dt_f = dt * 5
seed = 123
np.random.seed(seed)

N_train = 100
model_name =  f"NN_2layer_N{N_train}"      # Choose LinearRegression or NN 
model = NN(1, 1, [32, 32])
model_name = f"LinearRegression_N{N_train}"
# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_path = f'{data_path}/{model_name}/'

# Load ml param model
output_dicts = torch.load(f"{save_model_path}/model_best.pt")
ml_model = output_dicts["model"]
ml_model.eval()

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")
# Select initial conditions, separated by intervals of 10MTU 
sep = int(10/dt_f)
print(f"Initial conditions separated by {sep} time units")
X_init_conds = X_truth[::sep]
N_init = X_init_conds.shape[0]

# Initialize param_func
def param_func(X):
    nn_input = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        out = - ml_model(nn_input)
    return  out.squeeze().numpy()

# Run each model for 10MTU
T = 10
for i in range(N_init):
    # Initialize model
    l96_model = L96OneLayerParam(X_0=X_init_conds[i], param_func=param_func, dt=dt_f, F=F)

    # Run model
    X, U, time = l96_model.iterate(T)

    if i == 0:
        X_all = X
        U_all = U
    else:
        X_all = np.concatenate((X_all, X), axis=0)
        U_all = np.concatenate((U_all, U), axis=0)

# Save results
np.save(f"{save_model_path}/X_dtf.npy", X_all)
np.save(f"{save_model_path}/U_dtf.npy", U_all)

