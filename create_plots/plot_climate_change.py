import numpy as np
import matplotlib.pyplot as plt

import torch

from plotting_scripts.plot_distributions import plot_distributions


from plotting_scripts.plot_distributions import plot_distributions
from plotting_scripts.plot_ensemble_trajectories import plot_ensembles
from plotting_scripts.plot_rmse import plot_error_trajectories
from plotting_scripts.plot_regime_uncertainty_with_time import plot_regime_uncertainty_time
from plotting_scripts.plot_climate_properties import plot_climate_properties

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
F_tests = [16, 24]

model_name = f"BayesianNN_16_16_N{N_train}"
run_types = ["aleatoric_AR1", "epistemic_AR1", "both_AR1", "epistemic_fix", "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
label_names = [ "Aleatoric (AR1)", "Epistemic (AR1)", "Both (AR1)", "Epistemic (PPE)", "Both (PPE)"]
linestyles = ["solid"]*3 + ["dashed"]*3

for F_test in F_tests:
    save_prefix = f"climate_F{F_test}_"
    fnames = [f"climate_F{F_test}_run{i:02d}_X_dtf" for i in range(1)]
    fnames_true = [f"climate_F{F_test}_run{i:02d}_X_dtf" for i in range(1)]


    # All on one plot:
    plot_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles) 

    # Plot individual spaghetti plots:
    for i in range(1):
        save_prefix = f"climate_F{F_test}_run{i:02d}_"
        fname = f"climate_F{F_test}_run{i:02d}_X_dtf"
        for r in range(len(run_types)):
            plot_ensembles(params, model_name, run_types[r], label_names[r], 
                save_prefix=f'{save_prefix}{run_types[r]}_', fname=fname, 
                spaghetti=True, shading=False, save_step = 10, max_plots=0, xmax=5)


