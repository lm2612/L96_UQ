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
N_train = 100
dropout_rates = np.arange(0.1, 0.8, 0.1)

model_names =  [ 
    f"DropoutNN{dr:.1f}_2layer_N{N_train}/" for dr in dropout_rates]
epi_BNN = f"BayesianNN_2layer_N{N_train}/epistemic_"
colors = plt.cm.YlOrRd(np.linspace(0.1, 0.9,len(dropout_rates)))

# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_paths = [f'{data_path}/{model_name}' for model_name in model_names]
truth_path = f'{data_path}/truth/'
if len(model_names) == 1:
    plot_path = save_model_paths[0]
else:
    plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")

# Load BNN
X_BNN = np.load(f"{data_path}/{epi_BNN}X_dtf.npy")

# Load ml param model results
X_mls = [np.load(f"{save_model_path}X_dtf.npy") for save_model_path in save_model_paths]


T = np.ceil(X_truth.shape[0] * dt_f)
print(T, X_truth.shape,[X_ml.shape for X_ml in X_mls])
time = np.arange(0, T, dt_f)
print(time.shape)


# Separation timescales
T = 10
sep = int(T/dt_f)
print(f"Initial conditions separated by {sep} time units")
X_init_conds = X_truth[::sep]
N_init = X_init_conds.shape[0]
nt_total = X_truth.shape[0]

# Check 
nt = int(T/dt_f)
assert(N_init * nt == nt_total)


# Plot
plt.clf()
fig, axs = plt.subplots(3, 3, figsize=(20, 12)) #, sharex=True)
axs = axs.flatten()

# Choose 9 different initial conditions spaced out
# Fix j
j = 3

init_conds = [0, 4, 25, 
            31, 36, 46, 
            57, 86, 99 ]
for ii, i in enumerate(init_conds):
    axs[ii].plot(time[0:nt], X_truth[i*nt:(i+1)*nt, j],
        label="Truth", 
        alpha=1.,
        color="black")
    n_ens = X_BNN.shape[0]
    for n in range(n_ens):
        axs[ii].plot(time[0:nt], X_BNN[n, i*nt:(i+1)*nt, j],
            alpha=0.2,
            color="blue")
    for k, X_ml in enumerate(X_mls):
        n_ens = X_ml.shape[0]
        if n_ens > 1:
            for n in range(n_ens):
                axs[ii].plot(time[0:nt], X_ml[n, i*nt:(i+1)*nt, j],
                #label=labels[model_name] if n==0 else None, 
                alpha=0.1 if n_ens > 20 else 0.5,
                color=colors[k])
        if n_ens > 1:
            axs[ii].plot(time[0:nt], X_ml[:, i*nt:(i+1)*nt, j].mean(axis=0),
                label=f"Dropout {dropout_rates[k]:.1f}", 
                alpha=0.5,
                color=colors[k])
        if n_ens == 1:
            axs[ii].plot(time[0:nt], X_ml[0, i*nt:(i+1)*nt, j],
                label=f"Dropout {dropout_rates[k]:.1f}", 
                alpha=0.9,
                color=colors[k])
        
    axs[ii].axis(xmin=0, xmax=3)
    if  len(model_names) > 1:  
        axs[ii].legend(loc="upper left")
    axs[ii].set_ylabel(f"X_{j}")
    axs[ii].set_xlabel("Time")
    axs[ii].set_title(f"Example {ii}")
plt.tight_layout()
plt.savefig(f"{plot_path}X_dropout_compare.png")

print(f"Saved to {plot_path}X_dropout_compare.png")
plt.close()


# Plot
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
for k, X_ml in enumerate(X_mls):
    # Get spread across ensembles
    X_std = X_ml.std(axis=0)
    X_std = X_std.reshape((N_init, sep, K))
    # Take mean across all initial conditions (axis=0) and variables (axis=2)
    X_std = X_std.mean(axis=(0, 2))    
    axs.plot(time[0:nt], X_std, 
    label=f"Dropout {dropout_rates[k]:.1f}", 
                alpha=0.7,
                color=colors[k])
axs.axis(xmin=0, xmax=10)
axs.legend(loc="lower right")
axs.set_ylabel(f"Standard deviation in ensemble")
axs.set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{plot_path}/X_dropout_std.png")
print(f"Saved to {plot_path}/X_dropout_std.png")
