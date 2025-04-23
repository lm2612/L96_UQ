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
N_train = 50
model_names =  [ f"BayesianNN_2layer_N{N_train}/aleatoric_", 
#f"BayesianNN_2layer_N{N_train}/epistemic_",
#f"BayesianNN_2layer_N{N_train}/aleatoric_",
#f"AleatoricNN_2layer_N{N_train}/"
#f"DropoutNN_2layer_N{N_train}/"
]      # Choose LinearRegression or NN 

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

for i in range(N_init):
    # Plot
    plt.clf()
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axs = axs.flatten()

    for j in range(len(axs)):
        axs[j].plot(time[0:nt], X_truth[i*nt:(i+1)*nt, j],
            label="Truth", 
            alpha=1.,
            color=colors["Truth"])
        for X_ml, model_name in zip(X_mls, model_names):
            n_ens = X_ml.shape[0]
            for n in range(n_ens):
                axs[j].plot(time[0:nt], X_ml[n, i*nt:(i+1)*nt, j],
                #label=labels[model_name] if n==0 else None, 
                alpha=0.1 if n_ens > 20 else 0.5,
                color=colors[model_name])
            if n > 1:
                axs[j].plot(time[0:nt], X_ml[:, i*nt:(i+1)*nt, j].mean(axis=0),
                    label=labels[model_name], 
                    alpha=0.5,
                    color=colors[model_name])
        axs[j].axis(xmin=0, xmax=3)
        if  len(model_names) > 1:  
            axs[j].legend(loc="upper left")
        axs[j].set_ylabel(f"X_{j}")
        axs[j].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"{plot_path}X_ens_timeseries_{i}.png")

    print(f"Saved to {plot_path}X_ens_timeseries_{i}.png")
    plt.close()

# Distributions
fig = plt.figure(figsize=(10, 5))
X_domain = np.linspace(-25., 25., 80)
pdf_truth = kde_plot(X_truth[:], X_domain)
plt.plot(pdf_truth, color="black", label="Truth")
for X_ml, model_name in zip(X_mls, model_names):
    pdf = kde_plot(X_ml[:], X_domain)
    plt.plot(pdf, 
            color=colors[model_name], 
            label=model_name, 
            alpha=0.6)
plt.legend()
plt.savefig(f"{plot_path}X_ens_pdf.png")
print(f"Saved to {plot_path}X_ens_pdf.png")
