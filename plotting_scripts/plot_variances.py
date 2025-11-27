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
N_train = 100
model_names =  [ 
    #f"BayesianNN_2layer_N{N_train}/both_", 
   #f"BayesianNN_2layer_N{N_train}/both_",
    #f"NN_2layer_regime0_N{N_train}/longrun_",
    #f"BayesianNN_2layer_N{N_train}/aleatoric_AR1_",
        #f"BayesianNN_2layer_N{N_train}/epistemic_",
        
        f"BayesianNN_hetero_32_N{N_train}/both_fix_AR1_",
         f"BayesianNN_hetero_32_N{N_train}/epistemic_fix_",
        f"BayesianNN_hetero_32_N{N_train}/aleatoric_AR1_",

##f"BayesianNN_2layer_N{N_train}/deterministic_",
#f"AleatoricNN_2layer_N{N_train}/deterministic_",
#f"DropoutNN_2layer_N{N_train}/deterministic_",
#f"NN_2layer_N{N_train}/",
#f"AleatoricNN_2layer_N{N_train}/"
#f"DropoutNN_2layer_N{N_train}/"
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
if len(model_names) == 1:
    plot_path = save_model_paths[0]
else:
    plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
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




# Choose 9 different initial conditions spaced out
# Fix 
j = 3

init_conds = [0, 4, 25, 
            31, 36, 46, 
            57, 86, 99 ]
init_conds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


for i in init_conds:
    # Plot
    plt.clf()
    fig, axs = plt.subplots(4, 2, figsize=(14, 12)) 
    axs = axs.flatten()
    for j in range(8):
        for X_ml, model_name, label_name in zip(X_mls, model_names, label_names):
            n_ens = X_ml.shape[0]
            if n_ens > 1:
                # compute variance across ensemble 
                X_timeseries_ij = X_ml[:, i*nt:(i+1)*nt, j]
                variance_ij = np.var(X_timeseries_ij, axis=0)
                axs[j].plot(time[0:nt], variance_ij,
                    alpha=0.8, 
                    label=label_name, 
                    color=plotcolor(model_name),
                    lw=2)
                if label_name == "Epistemic":
                    variance_sum_ij = variance_ij
                elif label_name == "Aleatoric":
                    variance_sum_ij += variance_ij
                    axs[j].plot(time[0:nt], variance_sum_ij,
                        alpha=0.8, 
                        label="Sum Epistemic+Aleatoric", 
                        color="grey",
                        lw=2, linestyle="dashed")

        axs[j].axis(xmin=0, xmax=2., ymin=0., ymax=6)
        if  len(model_names) > 1:  
            axs[j].legend(loc="upper left")
        axs[j].set_ylabel(f"X_{j}")
        axs[j].set_xlabel("Time")
        axs[j].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{plot_path}X_variance_allvars_init_cond_{i}.png")

    print(f"Saved to {plot_path}X_variance_allvars_init_cond_{i}.png")
    plt.close()

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(7, 3)) 

    for X_ml, model_name, label_name in zip(X_mls, model_names, label_names):
        n_ens = X_ml.shape[0]
        if n_ens > 1:
            # compute variance across ensemble 
            X_timeseries_ij = X_ml[:, i*nt:(i+1)*nt, :]
            variance_ij = np.var(X_timeseries_ij, axis=(0)).mean(axis=1)
            ax.plot(time[0:nt], np.sqrt(variance_ij),
                alpha=0.8, 
                label=label_name, 
                color=plotcolor(model_name),
                lw=2)
            if label_name == "Epistemic":
                variance_sum_ij = variance_ij
            elif label_name == "Aleatoric":
                variance_sum_ij += variance_ij
                ax.plot(time[0:nt], np.sqrt(variance_sum_ij),
                    alpha=0.8, 
                    label="Sum Epistemic+Aleatoric", 
                    color="grey",
                    lw=2, linestyle="dashed")

        ax.axis(xmin=0, xmax=2., ymin=0., ymax=6)
        ax.set_ylabel(f"SQRT(Variance)")
        ax.set_xlabel("Time")
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{plot_path}X_variance_init_cond_{i}.png")

    print(f"Saved to {plot_path}X_variance_init_cond_{i}.png")
    plt.close()

# Reshape arrays to account for initial conditions
def reshape_by_init_conds(X, T = 10, N_init = 100):
    sep = int(T/dt_f)
    
    if X.ndim == 2:
        n_init = X.shape[0] // (sep)
        return X.reshape((n_init, sep, K))
    elif X.ndim == 3:
        n_init = X.shape[1] // (sep)
        return X.reshape((X.shape[0], n_init, sep, K))
# Reshape by
X_truth = reshape_by_init_conds(X_truth, N_init = N_init)
X_mls = [reshape_by_init_conds(X_ml, T=T, N_init = N_init) for X_ml in X_mls]
print(X_truth.shape, X_mls[0].shape)


# Plot
plt.clf()
fig, ax = plt.subplots(1,1, figsize=(8, 4)) 
for X_ml, model_name, label_name in zip(X_mls, model_names, label_names):
    variance = np.var(X_ml, axis=0).mean(axis=(0, 2))
    ax.plot(time[0:nt], np.sqrt(variance),
            alpha=0.8, 
            label=label_name, 
            color=plotcolor(model_name),
            lw=2)
    if label_name == "Epistemic":
        variance_sum = variance
    elif label_name == "Aleatoric":
        variance_sum += variance
        ax.plot(time[0:nt], np.sqrt(variance_sum),
            alpha=0.8, 
            label="Sum Epistemic+Aleatoric", 
            color="grey",
            lw=2, linestyle="dashed")
    #ax.set_yscale('log')
    ax.axis(xmin=0, xmax=2., ymin=0., ymax=8)
    if  len(model_names) > 1:  
        ax.legend(loc="upper left")
    ax.set_ylabel(f"Variance")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f"{plot_path}X_variance_all.png")

print(f"Saved to {plot_path}X_variance_all.png")
plt.close()
