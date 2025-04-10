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

# models to plot
N_train = 100
model_name =  f"LinearRegression_N{N_train}"      # Choose LinearRegression or NN 


# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_path = f'{data_path}/{model_name}/'
truth_path = f'{data_path}/truth/'

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")

# Load ml param model results
X_ml = np.load(f"{save_model_path}/X_dtf.npy")

T = np.ceil(X_truth.shape[0] * dt_f)
print(T, X_truth.shape, X_ml.shape)
time = np.arange(0, T, dt_f)
print(time.shape)

# Plot
max_time = 4000
plt.plot(time[:max_time], X_truth[:max_time, 0], label="Truth", alpha=0.5)
plt.plot(time[:max_time], X_ml[:max_time, 0], label="ML param", alpha=0.5)
plt.legend()
plt.savefig(f"{save_model_path}/X0.png")
print(f"Saved to {save_model_path}/X0.png")


