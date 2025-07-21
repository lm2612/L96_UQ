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
F = 20  
c = 10
b = 10

# Define time-stepping, random seed
dt = 0.001
dt_f = dt * 5
seed = 123
np.random.seed(seed)


# Set up model and types of simulations to plot
N_train = 50
model_name = f"BayesianNN_16_N{N_train}"
run_types = ["epistemic", "aleatoric", "both"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
colors = ["darkorchid", "seagreen", "dimgrey"]
label_names = [ "Epistemic", "Aleatoric", "Both"]
save_prefix = "whitenoise_"


# Set up directories
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
model_path = f'{data_path}/{model_name}/'
filenames = [f'{model_path}/{run_type}_X_dtf.npy' for run_type in run_types]
print(filenames)

plot_path = f'{model_path}/plots/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Load truth data
X_truth = np.load(f"{data_path}/X_dtf.npy")

# Load ml param model results
test_params = [np.load(f"{model_path}/{run_type}_test_params.npy", allow_pickle=True).item() for run_type in run_types]
X_mls = [np.load(filename) for filename in filenames]

# Get info about simulation - number of init conds, time T, number of ensembles
N_init = min([test_param['N_init'] for test_param in test_params])
T =  min([test_param['T'] for test_param in test_params])
n_ens =  min([test_param['n_ens'] for test_param in test_params])

print(T, X_truth.shape,[X_ml.shape for X_ml in X_mls])
time = np.arange(0, T, dt_f)
print(time.shape)

# Separation timescales
nt = int(T/dt_f)
print(f"Initial conditions separated by {nt} time units")
X_init_conds = X_truth[::nt]

for i in range(N_init):
    print(f"Plotting initial condition {i}")
    
    plt.clf()
    fig, axs = plt.subplots(4, 2, figsize=(14, 12)) 
    axs = axs.flatten()
    for j in range(8):
        axs[j].plot(time[0:nt], X_truth[i*nt:(i+1)*nt, j],
        label="Truth", 
        alpha=1.,
        color=plotcolor("Truth"))
        for X_ml, run_type, label_name in zip(X_mls, run_types, label_names):
            n_ens = X_ml.shape[0]
            if n_ens > 1:
                for n in range(n_ens):
                    axs[j].plot(time[0:nt], X_ml[n, i*nt:(i+1)*nt, j],
                    #label=labels[model_name] if n==0 else None, 
                    alpha=0.1 if n_ens > 20 else 0.5,
                    color=plotcolor(run_type))
            if n_ens > 1:
                axs[j].plot(time[0:nt], X_ml[:, i*nt:(i+1)*nt, j].mean(axis=0),
                    label=label_name, 
                    alpha=0.8, lw=2,
                    color=plotcolor(run_type))
            if n_ens == 1:
                axs[j].plot(time[0:nt], X_ml[0, i*nt:(i+1)*nt, j],
                    label=label_name, 
                    alpha=0.9,
                    color=plotcolor(run_type))
        axs[j].axis(xmin=0, xmax=2., ymin = -10., ymax=16.)
        axs[j].legend(loc="upper left")
        axs[j].set_ylabel(f"X_{j}")
        axs[j].set_xlabel("Time")
        axs[j].set_title(f"Initial Condition {i}")
        axs[j].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{plot_path}{save_prefix}ensemble_timeseries_{i}.png")
    print(f"Saved as {plot_path}{save_prefix}ensemble_timeseries_{i}.png")
    plt.close()

    