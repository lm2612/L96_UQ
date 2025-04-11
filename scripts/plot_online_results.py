import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

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

# Colors for plotting (keep this fixed)
colors = {"Truth": "black",
          "OneLayer":"gray",
          "LinearRegression_N100":"red",
           "NN_2layer_N100":"orange",
          }

# models to plot
N_train = 100
model_names =  [f"LinearRegression_N{N_train}", f"NN_2layer_N{N_train}"]      # Choose LinearRegression or NN 


# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_paths = [f'{data_path}/{model_name}/' for model_name in model_names]
truth_path = f'{data_path}/truth/'
plot_path = f'./plots/'
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

# Plot
max_time = 1000
fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
axs = axs.flatten()
for j in range(len(axs)):
    axs[j].plot(time[:max_time], X_truth[:max_time, j], label="Truth", alpha=0.5)
    for X_ml, model_name in zip(X_mls, model_names):
        axs[j].plot(time[:max_time], X_ml[:max_time, j], label=model_name, alpha=0.5)
    axs[j].legend(loc="upper left")
    axs[j].set_ylabel(f"X_{j}")
    axs[j].set_xlabel("Time")
plt.tight_layout()
plt.savefig(f"{plot_path}/X_timeseries.png")

print(f"Saved to {plot_path}/X_timeseries.png")

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
plt.savefig(f"{plot_path}/X_pdf.png")
print(f"Saved to {plot_path}/X_pdf.png")
