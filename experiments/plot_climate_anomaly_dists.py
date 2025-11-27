import numpy as np
import matplotlib.pyplot as plt

import torch

from plotting_scripts.plot_distributions import plot_distributions
from plotting_scripts.plot_runningmean_distributions import plot_runningmean_distributions
from plotting_scripts.plot_anomaly_distributions import plot_anomaly_distributions

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

F_tests = [16, 24]

run_types = ["aleatoric_AR1", "epistemic_AR1", "both_AR1", "epistemic_fix", "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
label_names = [ "Aleatoric", "Epistemic", "Both", "Epistemic (PPE)", "Both (PPE)"]
linestyles = ["solid"]*3 + ["dashed"]*3

for F_test in F_tests:
    save_prefix = f"climate_F{F_test}_"
    fnames = [f"climate_F{F_test}_run{i:02d}_X_dtf" for i in range(1)]
    fnames_true = [f"climate_F20_run{i:02d}_X_dtf" for i in range(1)]
    #base_name = 

    # All on one plot:
    plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles)

    plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles, running_mean=True, running_mean_window=4) 


    plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles, running_mean=True, running_mean_window=10) 
    
    plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles, running_mean=True, running_mean_window=40) 


    plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles, running_mean=True, running_mean_window=80)


    plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles, running_mean=True, running_mean_window=100)

    plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
        save_prefix=save_prefix, fnames=fnames, linestyles=linestyles, running_mean=True, running_mean_window=120) 