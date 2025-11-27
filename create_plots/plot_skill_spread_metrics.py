import numpy as np
import matplotlib.pyplot as plt

import torch

from plotting_scripts.plot_skill_spread_metrics import plot_spread_v_skill

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
model_name = f"BayesianNN_Heteroscedastic_16_16_N100"
run_types = ["aleatoric", "epistemic", "both"] 
label_names = ["Aleatoric (indep.)", "Epistemic (indep.)", "Both (indep.)"]
save_prefix = "Indep_"

plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    ylim=2., samples_per_bin=100)

run_types = ["aleatoric_AR1", "new_epistemic_AR1", "new_both_AR1"] 
label_names = ["Aleatoric (AR1)", "Epistemic (AR1)", "Both (AR1)"]
save_prefix = "AR1_"

plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    ylim=2., samples_per_bin=100)

