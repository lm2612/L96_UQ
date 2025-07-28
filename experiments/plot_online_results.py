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
N_train = 50
model_name = f"BayesianNN_16_N{N_train}"
run_types = ["aleatoric", "IC_aleatoric"] #, "IC_aleatoric", "aleatoric"] #, "aleatoric", "both"] #, "aleatoric_AR1", "both_fix_AR1"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
label_names = ["Aleatoric", "Aleatoric+InitCond"] #, "Aleatoric+InitCond", "Aleatoric"] #"Epistemic", "Aleatoric", "Both"]
save_prefix = "IC_"
run_types = ["offline_epistemic"]
label_names = ["Offline"]
save_prefix = "offline_"
#plot_distributions(params, model_name, run_types, label_names, save_prefix=save_prefix)
plot_ensembles(params, model_name, run_types, label_names, save_prefix=save_prefix, fname="X_dtf", spaghetti=False, shading=True)
#plot_error_trajectories(params, model_name, run_types, label_names, save_prefix=save_prefix)


