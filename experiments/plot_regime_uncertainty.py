import numpy as np
import matplotlib.pyplot as plt

import torch

from plotting_scripts.plot_regime_uncertainty_with_time import plot_regime_uncertainty_time


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
run_types = ["aleatoric_AR1", "epistemic_fix", "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
label_names = [ "Aleatoric (AR1)", "Epistemic (PPE)",  "Both"]
save_prefix = "AR1_"
fnames = [f"run{i:02d}_X_dtf" for i in range(1)]

#plot_regime_uncertainty_time(params, model_name, run_types, label_names, 
#    save_prefix=save_prefix, fnames = fnames, save_step=10)


Fs = [20, 16, 24]
for F in Fs:
    save_prefix = f"climate_F{F}"
    fnames = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(1)]

    plot_regime_uncertainty_time(params, model_name, run_types, label_names, 
        save_prefix=save_prefix, fnames = fnames, save_step=10)