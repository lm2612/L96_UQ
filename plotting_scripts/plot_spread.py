import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plot_dicts import colors, labels, plotcolor

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
N_train = 50
model_names =  [
    #"OneLayer/", 
 #f"LinearRegression_N{N_train}/",
 #f"NN_2layer_N{N_train}/",
 #f"DropoutNN_2layer_N{N_train}/",
 #f"DropoutNN0.8_2layer_N{N_train}/",
 #f"DropoutNN0.2_2layer_N{N_train}/",
 #f"AleatoricNN_2layer_N{N_train}/",
#f"BayesianNN_2layer_N{N_train}/epistemic_",
 f"BayesianNN_multivariatefull_2layer_N{N_train}/both_", 
f"BayesianNN_multivariatefull_2layer_N{N_train}/epistemic_", 
 f"BayesianNN_multivariatefull_2layer_N{N_train}/aleatoric_", 

#f"BayesianNN_2layer_N{N_train}/both_", 
#f"BayesianNN_2layer_N{N_train}/deterministic_", 

 ]      # Choose LinearRegression or NN 
label_names = [ 
"Both",
"Epistemic",
"Aleatoric"
]


# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_paths = [f'{data_path}/{model_name}' for model_name in model_names]
truth_path = f'{data_path}/truth/'
plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")

# Load ml param model results
X_mls = [np.load(f"{save_model_path}X_dtf.npy") for save_model_path in save_model_paths]


T = np.ceil(X_truth.shape[0] * dt_f)
print(T, X_truth.shape, X_mls[0].shape)
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


# Plot
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
for X_ml, model_name, label_name in zip(X_mls, model_names, label_names):
    # Get variance across ensembles
    X_var = X_ml.var(axis=0)
    X_var = X_var.reshape((N_init, sep, K))
    # Take mean across all initial conditions (axis=0) and variables (axis=2)
    X_var = X_var.mean(axis=(0, 2))   
    X_std = np.sqrt(X_var) 
    axs.plot(time[0:nt], X_std, 
    label=label_name, 
    alpha=0.8,
    color=plotcolor(model_name))
axs.axis(xmin=0, xmax=10)
axs.legend(loc="lower right")
axs.set_ylabel(f"Standard deviation in ensemble")
axs.set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{plot_path}/X_std_timeseries.png")
print(f"Saved to {plot_path}/X_std_timeseries.png")

# Plot
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
for X_ml, model_name, label_name in zip(X_mls, model_names, label_names):
    # Take mean prediction across ensembles (only relevant if n_ens > 1)
    X_mean = X_ml.mean(axis=0)
    X_mean = X_mean.reshape((N_init, sep, K))
    # Take rmse across all initial conditions (axis=0) and variables (axis=2)
    X_diff_squared = ((X_mean - X_truth)**2).mean(axis=(0, 2))
    X_rmse = np.sqrt(X_diff_squared)
    axs.plot(time[0:nt], X_rmse, 
    label=label_name + " RMSE", 
    alpha=0.8,
    color=plotcolor(model_name))
    
    # Get variance across ensembles
    X_var = X_ml.var(axis=0)
    X_var = X_var.reshape((N_init, sep, K))
    # Take mean across all initial conditions (axis=0) and variables (axis=2)
    X_var = X_var.mean(axis=(0, 2))   
    X_std = np.sqrt(X_var) 
    axs.plot(time[0:nt], X_std, 
    label=label_name+" STD", 
    linestyle="dashed",
    alpha=0.8,
    color=plotcolor(model_name))
axs.axis(xmin=0, xmax=6)
axs.legend(loc="lower right")
#axs.set_ylabel(f"")
axs.set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{plot_path}/X_spread_err_timeseries.png")
print(f"Saved to {plot_path}/X_spread_err_timeseries.png")

