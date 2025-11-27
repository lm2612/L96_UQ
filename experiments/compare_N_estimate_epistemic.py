import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression


from plotting_scripts.plot_distributions import plot_distributions
from plotting_scripts.plot_ensemble_trajectories import plot_ensembles
from plotting_scripts.plot_rmse import plot_error_trajectories

# Set up parameters for simulation
params ={
    'F': 20,
    'c': 10,
    'b': 10,
    'h': 1,
    'J': 32,
    'K': 8,
    'dt': 0.001,
    'dt_f': 0.005,
}

# Set up model and types of simulations to plot
N_train = 100
model_name = f"BayesianNN_16_16_N{N_train}"
Ns = [1, 2, 4, 5, 10, 20, 50]
run_types = [f"newN{N}_epistemic_AR1" for N in Ns] 
label_names = [f"N={N}" for N in Ns] 
linestyles = ["dotted", "dashed", "dashdot", "solid"]
colors = ["#2596be", "#2187ab", "#1e7898", "#1a6985", "#165a72", "#134b5f", "#0f3c4c"]
save_prefix = "newN"


plot_error_trajectories(params, model_name, run_types, label_names, save_prefix=save_prefix, colors=colors)