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

# This script plots the majority of the online results and extra that aren't in the paper

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
model_name = f"BayesianNN_16_16_N100_priorNormal(0,1.0)"

# First, white noise / independent noise simulations
#kernel_name = "NUTS"
training_method = "VI" # VA or mcmc_{kernel_name}
noise_type = "WN"

run_types = [f"{training_method}_{noise_type}_epistemic", 
   f"{training_method}_{noise_type}_aleatoric", 
    f"{training_method}_{noise_type}_both"]
label_names = ["Epistemic (indep.)", "Aleatoric (indep)", "Both (indep)"]
save_prefix = f"{training_method}_{noise_type}_"
for r in range(len(run_types)):
    plot_ensembles(params, model_name, run_types[r], label_names[r], 
        save_prefix=f'{save_prefix}{run_types[r]}_', fname="X_dtf", 
        spaghetti=True, shading=False, max_plots=1)

# Plot all in one plot:
plot_ensembles(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, fname="X_dtf", 
    spaghetti=False, shading=True, max_plots=1)

# Plot errors
plot_error_trajectories(params, model_name, run_types, label_names,
    save_prefix=save_prefix, include_sum=True, plot_spread = False)

# With spread as well (This is Fig.6a in paper)
plot_error_trajectories(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, include_sum=False, plot_spread = True)

# Plot distributions
plot_distributions(params, model_name, run_types, 
    label_names, save_prefix=save_prefix)

## AR1 
noise_type = "AR1"
run_types = [f"{training_method}_{noise_type}_epistemic", 
    f"{training_method}_{noise_type}_aleatoric", 
    f"{training_method}_{noise_type}_both"]
label_names = ["Epistemic (AR1)", "Aleatoric (AR1)", "Both (AR1)"]
save_prefix = f"{training_method}_{noise_type}_"
for r in range(len(run_types)):
    plot_ensembles(params, model_name, run_types[r], label_names[r], 
        save_prefix=f'{save_prefix}{run_types[r]}_', fname="X_dtf", 
        spaghetti=True, shading=False, max_plots=1)

# Plot all in one plot:
plot_ensembles(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, fname="X_dtf", 
    spaghetti=False, shading=True, max_plots=1)

# Plot errors
plot_error_trajectories(params, model_name, run_types, label_names,
    save_prefix=save_prefix, include_sum=True, plot_spread = False)

# With spread as well (This is Fig.6a in paper)
plot_error_trajectories(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, include_sum=False, plot_spread = True)

# Plot distributions
plot_distributions(params, model_name, run_types, 
    label_names, save_prefix=save_prefix)


