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
model_names =  [
 #f"DropoutNN_2layer_N{N_train}/",
 #f"AleatoricNN_2layer_N{N_train}/",
  f"BayesianNN_multivariate_2layer_N{N_train}/both_", 

 f"BayesianNN_multivariate_2layer_N{N_train}/epistemic_",
 f"BayesianNN_multivariate_2layer_N{N_train}/aleatoric_", 
# f"BayesianNN_2layer_N{N_train}/both_", 
 #f"BayesianNN_2layer_N{N_train}/deterministic_", 

 ]      # Stochastic models only

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

# Plot
for i in range(N_init):
    continue
    # Plot
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
    for X_ml, model_name in zip(X_mls, model_names):
        n_ens = X_ml.shape[0]
        for n in range(n_ens):
            axs.plot(time[0:nt], np.abs(X_ml[n, i*nt:(i+1)*nt] - X_truth[i*nt:(i+1)*nt]).mean(axis=-1),
            label=labels[model_name] if n==0 else None, 
            alpha=0.3 if n_ens > 20 else 0.5,
            color=colors[model_name])
    axs.axis(xmin=0, xmax=3)
    if  len(model_names) > 1:  
        axs.legend(loc="upper left")
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    axs.axhline(0, color="black", linestyle="dashed")
    plt.tight_layout()
    plt.savefig(f"{plot_path}X_diff_timeseries_{i}.png")

    print(f"Saved to {plot_path}X_diff_timeseries_{i}.png")
    plt.close()


# Reshape array 
print(sep, N_init, X_truth.shape)
X_truth = X_truth.reshape((N_init, sep, K))
print(X_truth.shape)
X_mls = [X_ml.reshape((X_ml.shape[0], N_init, sep, K)) for X_ml in X_mls]

# Plot
for i in range(N_init):
    continue
    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
    for X_ml, model_name in zip(X_mls, model_names):
        n_ens = X_ml.shape[0]
        X_abs_diff = np.abs(X_ml[:, i] - X_truth[i]) 
        X_mean = X_abs_diff.mean(axis=0).mean(axis=-1)
        
        axs.plot(time[0:nt], X_mean,
            label=labels[model_name] ,
            alpha=0.8,
            color=colors[model_name])
        if X_abs_diff.shape[0] > 1:
            X_std = X_abs_diff.std(axis=0).mean(axis=-1)
            axs.fill_between(time[0:nt], X_mean-X_std, X_mean+X_std,
                alpha=0.1,
                color=colors[model_name])
    axs.axis(xmin=0, xmax=3)
    if  len(model_names) > 1:  
        axs.legend(loc="upper left")
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    axs.axhline(0, color="black", linestyle="dashed")
    plt.tight_layout()
    plt.savefig(f"{plot_path}X_meandiff_spread_timeseries_{i}.png")

    print(f"Saved to {plot_path}X_meandiff_spread_timeseries_{i}.png")

# Plot
for i in range(N_init):
    continue
    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
    for X_ml, model_name in zip(X_mls, model_names):
        n_ens = X_ml.shape[0]
        X_abs_diff = np.abs(X_ml[:, i] - X_truth[i]) 
        X_mean = X_abs_diff.mean(axis=0).mean(axis=-1)
        
        axs.plot(time[0:nt], X_mean,
            label=labels[model_name] ,
            alpha=0.8,
            color=colors[model_name])
    axs.axis(xmin=0, xmax=3)
    if  len(model_names) > 1:  
        axs.legend(loc="upper left")
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    axs.axhline(0, color="black", linestyle="dashed")
    plt.tight_layout()
    plt.savefig(f"{plot_path}X_meandiff_timeseries_{i}.png")

    print(f"Saved to {plot_path}X_meandiff_timeseries_{i}.png")

# Plot
for i in range(N_init):
    continue
    # Plot
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
    for X_ml, model_name in zip(X_mls, model_names):
        n_ens = X_ml.shape[0]
        X_abs_diff = np.abs(X_ml[:, i] - X_truth[i]) 
        if X_abs_diff.shape[0] > 1:
            X_std = X_abs_diff.std(axis=0).mean(axis=-1)
            axs.plot(time[0:nt], X_std,
                label=labels[model_name],
                alpha=0.8,
                color=colors[model_name])
    axs.axis(xmin=0, xmax=3)
    if  len(model_names) > 1:  
        axs.legend(loc="upper left")
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    axs.axhline(0, color="black", linestyle="dashed")
    plt.tight_layout()
    plt.savefig(f"{plot_path}X_spread_timeseries_{i}.png")

    print(f"Saved to {plot_path}X_spread_timeseries_{i}.png")


err_threshold = 1.
std_threshold = 1.
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)



for i in range(N_init):
    continue
    # Plot
    for X_ml, model_name in zip(X_mls, model_names):
        n_ens = X_ml.shape[0]
        if n_ens == 1:
            continue
        X_abs_diff = np.abs(X_ml[:, i] - X_truth[i]) 
        X_mean = X_abs_diff.mean(axis=0).mean(axis=-1)
        time_err = time[np.argwhere(X_mean > err_threshold)[0]]

        X_std = X_abs_diff.std(axis=0).mean(axis=-1)
        time_std = time[np.argwhere(X_std > std_threshold)[0]]

        axs.scatter(time_err, time_std,
            label=labels[model_name] if i == 0 else None,
            alpha=0.8,
            color=colors[model_name])
if  len(model_names) > 1:  
    axs.legend(loc="upper left")
axs.set_ylabel(f"Time until std > {std_threshold}")
axs.set_xlabel(f"Time until err > {err_threshold}")
axs.plot([0., 6.], [0., 6.], color="black", linestyle="dashed")
plt.tight_layout()
plt.savefig(f"{plot_path}time_until_divergence.png")


plt.clf()
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
T = 0.6
t = int(T/dt_f)
print(t)
for i in range(4):
    # Plot
    for X_ml, model_name in zip(X_mls, model_names):
        n_ens = X_ml.shape[0]
        if n_ens == 1:
            continue
        X_abs_diff = np.abs(X_ml[:, i, t] - X_truth[i, t])
        X_mae = X_abs_diff.mean(axis=0)
        X_rmse = np.sqrt((X_abs_diff**2).mean(axis=0)).flatten()
        mean_err = np.abs(X_ml[:, i, t].mean(axis=0) - X_truth[i, t]).flatten()
        X_std = X_ml[:, i, t].std(axis=0).flatten()

        axs.scatter(X_mae, X_std,
            label=labels[model_name] if i == 0 else None,
            alpha=0.8,
            color=colors[model_name])
if  len(model_names) > 1:  
    axs.legend(loc="upper left")
axs.set_ylabel(f"Spread at time {T}")
axs.set_xlabel(f"Error in mean at time {T}")
axs.plot([0., 6.], [0., 6.], color="black", linestyle="dashed")
plt.tight_layout()
plt.savefig(f"{plot_path}error_v_spread_scatter_t{T}.png")
print(f"{plot_path}error_v_spread_scatter_t{T}.png")

plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(12, 4))


domain = np.linspace(0., 6., 50)

for X_ml, model_name in zip(X_mls, model_names):
    n_ens = X_ml.shape[0]
    if n_ens == 1:
        continue
    time_errs = []
    time_stds = []
    for i in range(N_init): 
        X_abs_diff = np.abs(X_ml[:, i] - X_truth[i]) 
        X_mean = X_abs_diff.mean(axis=0).mean(axis=-1)
        time_errs.append(time[np.argwhere(X_mean > err_threshold)[0]][0])

        X_std = X_abs_diff.std(axis=0).mean(axis=-1)
        time_stds.append(time[np.argwhere(X_std > std_threshold)[0]][0])

    axs[0].plot(domain, kde_plot(time_errs, domain), 
    alpha=0.6,
    color = colors[model_name],
    label = labels[model_name])

    axs[1].plot(domain, kde_plot(time_stds, domain), 
    alpha=0.6,
    color = colors[model_name],
    label = labels[model_name])

axs[0].set_title(f"Time until prediction diverges from truth by {err_threshold}")
axs[0].set_xlabel(f"Time (MTU) ")
axs[0].set_ylabel(f"PDF")
axs[0].legend(loc="upper right")

axs[1].set_title(f"Time until ensemble spread exceeds {std_threshold}")
axs[1].set_xlabel(f"Time (MTU) ")
axs[1].set_ylabel(f"PDF")
axs[1].legend(loc="upper right")

plt.savefig(f"{plot_path}pdf_time_until_divergence.png")


