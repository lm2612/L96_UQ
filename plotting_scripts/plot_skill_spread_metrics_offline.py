import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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

# Number of initial conditions and time ran for
N_init = 50 #100
T = 10

# models to plot
N_train = 100
model_names =  [

  f"BayesianNN_hetero_32_N{N_train}/offline_aleatoric_", 
  f"BayesianNN_hetero_32_N{N_train}/offline_epistemic_", 
  f"BayesianNN_hetero_32_N{N_train}/offline_both_", 


 ]     # Choose LinearRegression or NN 


# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_paths = [f'{data_path}/{model_name}' for model_name in model_names]
truth_path = f'{data_path}/truth/'
plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
if not os.path.exists(plot_path):
    os.makedirs(plot_path)

# Load truth data
spinup = 0
max_T = 2001
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")[spinup+1:max_T+1]

# Load ml param model results
print("Opening: ", [f"{save_model_path}X_dtf.npy" for save_model_path in save_model_paths] )
X_mls = [np.load(f"{save_model_path}X_dtf.npy")[:, spinup:max_T] for save_model_path in save_model_paths]

print(X_truth.shape, X_mls[0].shape)

# Ensemble mean
X_means = [X_ml.mean(axis=0) for X_ml in X_mls]
# Compute error
X_diffs = [(X_mean - X_truth) for X_mean in X_means]

# Compute spread
X_stds = [X_ml.std(axis=0) for X_ml in X_mls]

model_names = [ "aleatoric", "epistemic", "both"]
cmaps = [ "Greens", "Purples", "Blues"]
# Basic error v spread plot
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
for X_diff, X_std, model_name, cmap in zip(X_diffs, X_stds, model_names, cmaps):
    # Flatten
    X_diff = X_diff.flatten()
    X_std = X_std.flatten()
    print(X_diff.shape, X_std.shape)

    err = np.abs(X_diff)
    # Calculate the point density for colors
    xy = np.vstack([err, X_std])
    z = stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    #err, sigma, z = errs[idx], sigma[idx], z[idx]
    plt.scatter(err, X_std, label=model_name, alpha=0.6, c=z, cmap=cmap, edgecolor=None)
# Add y = x line
ylim = np.max(err) * 0.8
plt.plot([0, ylim], [0, ylim], 'k--')
plt.xlabel("Absolute Error")
plt.ylabel("Standard Deviation")
plt.axis(xmin=0, xmax=ylim, ymin=0, ymax=ylim)
plt.legend()
plt.savefig(f"{plot_path}/offline_X_abs_err_v_std.png")
print(f"Saved to {plot_path}/offline_X_abs_err_v_std.png")

# Now sort
plt.clf()
n_samples = X_truth.shape[0]*K
samples_per_bin = 8
n_bins = n_samples // samples_per_bin
print(samples_per_bin, n_bins, samples_per_bin*n_bins, X_truth.shape[0])
fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
cmaps = [ "Greens", "Purples", "Blue"]
colors = ["seagreen", "darkorchid", "dimgrey"]
for X_diff, X_std, model_name, color in zip(X_diffs, X_stds, model_names, colors):
    # Flatten
    print(X_diff.shape)
    X_diff = X_diff.flatten()
    X_std = (X_std**2).flatten()
    
    print(X_diff.shape, X_std.shape)

    # Sort into order of increasing spread
    sorted_inds = np.argsort(X_std)
    X_diff_sorted = (X_diff**2)[sorted_inds]
    X_std_sorted = X_std[sorted_inds]

    # Reshape into bins
    X_diff_bins = X_diff_sorted.reshape((n_bins, samples_per_bin))
    X_std_bins = X_std_sorted.reshape((n_bins, samples_per_bin))

    # Average across stds and take std of err
    spread = np.sqrt(X_std_bins.mean(axis=-1))
    sigma_err = np.sqrt(X_diff_bins.std(axis=-1))

    plt.scatter(spread, sigma_err, color=color, label=model_name, alpha=0.25)
plt.legend(loc="lower right")
plt.xlabel("r.m.s spread")
plt.ylabel("r.m.s error")
ylim=0.5

plt.plot([0, ylim], [0, ylim], 'k--')
plt.axis(xmin=0, xmax=ylim, ymin=0, ymax=ylim)



plt.savefig(f"{plot_path}/offline_spread_v_sigma_err.png")
print(f"Saved to {plot_path}/offline_spread_v_sigma_err.png")    

