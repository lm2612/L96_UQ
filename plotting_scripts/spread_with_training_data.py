import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plot_dicts import colors, labels

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
N_trains = [10, 20, 30, 40, 50, 60, 100, ]

# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
truth_path = f'{data_path}/truth/'
plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")

T = np.ceil(X_truth.shape[0] * dt_f)
time = np.arange(0, T, dt_f)
print(time.shape)


# Separation timescales
T = 10
sep = int(T/dt_f)
print(f"Initial conditions separated by {sep} time units")
X_init_conds = X_truth[::sep]
N_init = X_init_conds.shape[0]
nt_total = X_truth.shape[0]
print(X_truth.shape)
# Check 
nt = int(T/dt_f)
assert(N_init * nt == nt_total)

# Reshape array 
print(sep, N_init, X_truth.shape)
X_truth = X_truth.reshape((N_init, sep, K))
print(X_truth.shape)
X_truth = X_truth[:4]

def spread(X):
    v = X.var(axis=0).reshape((N_init, sep, K))[:4]
    # Take mean across all initial conditions (axis=0) and variables (axis=2)
    v = v.mean(axis=(0, 2))   
    std = np.sqrt(v) 
    return std

t = int( 0.6/dt_f )
# Load ml param model results
for N_train in N_trains:
    epi_model_path = f'{data_path}/BayesianNN_2layer_N{N_train}/epistemic_' 
    ale_model_path = f'{data_path}/BayesianNN_2layer_N{N_train}/aleatoric_' 
    both_model_path = f'{data_path}/BayesianNN_2layer_N{N_train}/both_' 
    X_epi = np.load(f"{epi_model_path}X_dtf.npy")
    X_ale = np.load(f"{ale_model_path}X_dtf.npy")
    X_both = np.load(f"{both_model_path}X_dtf.npy") 

    # Get spread across ensembles
    spread_epi = spread(X_epi) 
    spread_ale = spread(X_ale)
    spread_both = spread(X_both)
    plt.bar(N_train*8, spread_both[t], color="blue", width=40, label="total", alpha=0.8)
    plt.bar(N_train*8, spread_ale[t], color="seagreen", width=40, label="aleatoric", alpha=0.8)
    plt.bar(N_train*8, spread_epi[t], color="darkorchid",  width=40, label="epistemic", alpha=0.8)
    plt.xlabel("Number of training data points")
    plt.ylabel(f"Spread at time 0.6 MTU")

    if N_train ==N_trains[0]:
        plt.legend()

plt.savefig(f"{plot_path}/spread_v_N.png")
print(f"Saved to {plot_path}/spread_v_N.png")

def diff(X):
    X_m = X.mean(axis=0).reshape((N_init, sep, K))[:4]
    # Mean error in ensemble mean across all initial conditions (axis=0) and variables (axis=2)
    diff = ((X_m - X_truth)**2).mean(axis=(0, 2))   
    diff = np.sqrt(diff) 
    return diff
plt.clf()
# Load ml param model results
for N_train in N_trains:
    epi_model_path = f'{data_path}/BayesianNN_2layer_N{N_train}/epistemic_' 
    ale_model_path = f'{data_path}/BayesianNN_2layer_N{N_train}/aleatoric_' 
    both_model_path = f'{data_path}/BayesianNN_2layer_N{N_train}/both_' 
    X_epi = np.load(f"{epi_model_path}X_dtf.npy") 
    X_ale = np.load(f"{ale_model_path}X_dtf.npy") 
    X_both = np.load(f"{both_model_path}X_dtf.npy") 

    # Get spread across ensembles
    diff_epi = diff(X_epi) 
    diff_ale = diff(X_ale)
    diff_both = diff(X_both)
    plt.bar(N_train*8, diff_both[t], color="blue", width=40, label="total", alpha=0.8)
    plt.bar(N_train*8, diff_ale[t], color="seagreen", width=40, label="aleatoric", alpha=0.8)
    plt.bar(N_train*8, diff_epi[t], color="darkorchid",  width=40, label="epistemic", alpha=0.8)
    plt.xlabel("Number of training data points")
    plt.ylabel(f"RMSE in ensemble mean at time 0.6 MTU")

    if N_train ==N_trains[0]:
        plt.legend()

plt.savefig(f"{plot_path}/err_v_N.png")
print(f"Saved to {plot_path}/err_v_N.png")