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
N_train = 100
model_name = f"BayesianNN_Heteroscedastic_16_16_N{N_train}"

## First, white noise / independent noise simulations
run_types = ["aleatoric", "epistemic", "both"]
label_names = ["Aleatoric (indep.)", "Epistemic (indep.)", "Both (indep.)"] 
save_prefix = "Indep_"

# Plot individual spaghetti plots:
for r in range(len(run_types)):
    plot_ensembles(params, model_name, run_types[r], label_names[r], 
        save_prefix=f'{save_prefix}{run_types[r]}_', fname="X_dtf", 
        spaghetti=True, shading=False, max_plots=10)

# Plot all in one plot:
plot_ensembles(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, fname="X_dtf", 
    spaghetti=False, shading=True, max_plots=10)

# Plot distributions
plot_distributions(params, model_name, run_types, 
label_names, save_prefix=save_prefix)

# And error growth
plot_error_trajectories(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, include_sum=True)
# With spread as well (This is Fig.6a in paper)
plot_error_trajectories(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, include_sum=False, plot_spread = True)


# Next plot AR1 
run_types = ["aleatoric_AR1", "new_epistemic_AR1", "new_both_AR1"] 
label_names = ["Aleatoric (AR1)", "Epistemic (AR1)", "Both (AR1)"] 
save_prefix = "AR1_"

# Plot individual spaghetti plots:
for r in range(len(run_types)):
    plot_ensembles(params, model_name, run_types[r], label_names[r], 
        save_prefix=f'{save_prefix}{run_types[r]}_', fname="X_dtf", 
        spaghetti=True, shading=False, max_plots=10)

# Plot all
plot_ensembles(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, fname="X_dtf", 
    spaghetti=False, shading=True, max_plots=10)


plot_error_trajectories(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    include_sum=True, plot_spread = False)
# Error / spread growth - Fig 6b. in paper
plot_error_trajectories(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    include_sum=False, plot_spread = True)


# Plot distributions with all in there:
run_types = ["aleatoric", "epistemic", "both",
            "aleatoric_AR1", "epistemic_AR1", "both_AR1"]
label_names = ["Aleatoric", "Epistemic", "Both",
            "Aleatoric (AR1)", "Epistemic (AR1)", "Both (AR1)"] #, "Deterministic"] #, "Aleatoric+InitCond", "Aleatoric"] #"Epistemic", "Aleatoric", "Both"]
linestyles = ["solid", "solid", "solid", "dashed", "dashed", "dashed"]
save_prefix = "all_"

plot_distributions(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, linestyles=linestyles)

