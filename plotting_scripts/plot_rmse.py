import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot
from utils.crps import crps


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
N_train = 100
model_names =  [
    #"OneLayer/", 
# f"LinearRegression_N{N_train}/",
# f"NN_2layer_N{N_train}/",
# f"DropoutNN_2layer_N{N_train}/",
# f"AleatoricNN_2layer_N{N_train}/",
 f"BayesianNN_2layer_N{N_train}/epistemic_",
 f"BayesianNN_2layer_N{N_train}/aleatoric_",  
 f"BayesianNN_2layer_N{N_train}/both_",  
# f"BayesianNN_2layer_N{N_train}/deterministic_", 
 ]      # Choose LinearRegression or NN 


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
for X_ml, model_name in zip(X_mls, model_names):
    # Take mean prediction across ensembles (only relevant if n_ens > 1)
    X_ml = X_ml.mean(axis=0)
    X_ml = X_ml.reshape((N_init, sep, K))
    # Take rmse across all initial conditions (axis=0) and variables (axis=2)
    X_diff = np.sqrt(((X_ml - X_truth)**2).mean(axis=(0, 2)))

    axs.plot(time[0:nt], X_diff, 
    label=labels[model_name], 
    alpha=0.8,
    color=colors[model_name])
axs.axis(xmin=0, xmax=10)
axs.legend(loc="lower right")
axs.set_ylabel(f"X")
axs.set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{plot_path}/X_rmse_timeseries.png")

print(f"Saved to {plot_path}/X_rmse_timeseries.png")

# Plot
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
for X_ml, model_name in zip(X_mls, model_names):
    # Get spread in err across ensembles
    X_std = ((X_ml.reshape(X_ml.shape[0], N_init, sep, K) - X_truth)**2).std(axis=0)
    X_std = X_std.reshape((N_init, sep, K))
    # Take mean across all initial conditions (axis=0) and variables (axis=2)
    X_std = np.sqrt(X_std.mean(axis=(0, 2))    )

    # Take mean prediction across ensembles (only relevant if n_ens > 1)
    X_ml = X_ml.mean(axis=0)
    X_ml = X_ml.reshape((N_init, sep, K))
    # Take rmse across all initial conditions (axis=0) and variables (axis=2)
    X_diff = np.sqrt(((X_ml - X_truth)**2).mean(axis=(0, 2))) 

    axs.plot(time[0:nt], X_diff, 
    label=labels[model_name], 
    alpha=0.8,
    color=colors[model_name])

    axs.fill_between(time[0:nt], X_diff-X_std,  X_diff + X_std,
    alpha=0.2,
    color=colors[model_name])
axs.axis(xmin=0, xmax=10)
axs.legend(loc="lower right")
axs.set_ylabel(f"X")
axs.set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{plot_path}/X_rmse_spread_timeseries.png")

print(f"Saved to {plot_path}/X_rmse_spread_timeseries.png")



# Plot
plt.clf()
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
for X_ml, model_name in zip(X_mls, model_names):
    n_ens = X_ml.shape[0]
    if n_ens > 1:
        X_ml = X_ml.reshape((n_ens, N_init, sep, K))
        err = crps(X_truth, X_ml).mean(axis=-1)
        axs.plot(time[0:nt], err, 
        label=labels[model_name], 
        alpha=0.8,
        color=colors[model_name])
axs.axis(xmin=0, xmax=4)
axs.legend(loc="lower right")
axs.set_ylabel(f"X")
axs.set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{plot_path}/X_crps_timeseries.png")

print(f"Saved to {plot_path}/X_crps_timeseries.png")