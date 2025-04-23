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
model_names =  ["OneLayer", 
 f"LinearRegression_N{N_train}",
 f"NN_2layer_N{N_train}", 
 ]      # Choose LinearRegression or NN 


# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_paths = [f'{data_path}/{model_name}/' for model_name in model_names]
truth_path = f'{data_path}/truth/'
plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")

# Load ml param model results
X_mls = [np.load(f"{save_model_path}/X_dtf.npy") for save_model_path in save_model_paths]


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

# Check 
nt = int(T/dt_f)
assert(N_init * nt == nt_total)

def lorenz_stats(X):
    # Calculate statistics for comparison to true statistics
    # Mean and std
    X_mean = X.mean()
    X_std = X.std()

    # Calculate peak wavenumber
    X_fft = np.fft.fft(X, axis=1)
    wavelengths = np.fft.fftfreq(K)
    X_fft = np.abs(X_fft)
    wavelengths, X_fft = wavelengths[wavelengths>0], X_fft[:, wavelengths>0]
    peak_ind = np.argmax(X_fft, axis=1)
    peak_wavelengths = np.array([wavelengths[i] for i in peak_ind])
    peak_wavenumber = 1./np.mean(peak_wavelengths)

    # Calculate rotation period (fft in time)
    X_fft = np.fft.fft(X, axis=0)
    freqs = np.fft.fftfreq(X.shape[0], d=dt)
    X_fft = np.abs(X_fft)
    # Get positive values only
    freqs, X_fft = freqs[freqs>0], X_fft[freqs>0]
    peak_ind = np.argmax(X_fft, axis=0)
    peak_freqs = np.array([freqs[i] for i in peak_ind])
    peak_period = 1./np.mean(peak_freqs)

    return np.array([X_mean, X_std, peak_wavenumber, peak_period])

for i in range(N_init):
    # Plot
    fig, axs = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axs = axs.flatten()
    # Save for each initial condition
    for j in range(len(axs)):
        stat_i =  lorenz_stats(X_truth[i*nt:(i+1)*nt])
        axs[j].scatter(np.arange(len(stat_i)), stat_i, 
            label=labels["Truth"], 
            alpha=1.,
            color=colors["Truth"],
            marker = "o")
        for X_ml, model_name in zip(X_mls, model_names):
            stat_i =  lorenz_stats(X_ml[0, i*nt:(i+1)*nt])
        
            axs[j].scatter(np.arange(len(stat_i)), stat_i, 
            label=labels[model_name], 
            alpha=0.5,
            color=colors[model_name],
            marker = "o")
        #axs[j].axis(xmin=0, xmax=3)
        axs[j].legend(loc="upper left")
        #axs[j].set_ylabel(f"X_{j}")
        #axs[j].set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/X_stats_{i}.png")

    print(f"Saved to {plot_path}/X_stats_{i}.png")

