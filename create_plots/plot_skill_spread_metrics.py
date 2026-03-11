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
model_name = f"BayesianNN_Heteroscedastic_16_16_N100_priorNormal(0,1.0)"

# First, white noise / independent noise simulations
training_method = "VI" # VI or mcmc_{kernel_name}
noise_type = "WN"

run_types = [f"{training_method}_{noise_type}_epistemic", 
    f"{training_method}_{noise_type}_aleatoric", 
    f"{training_method}_{noise_type}_both"]
label_names = ["Epistemic (indep.)", 
"Aleatoric (indep)", 
"Both (indep)"]
save_prefix = f"{training_method}_{noise_type}_"
    

plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    ylim=2, samples_per_bin=100)
noise_type = "AR1"
run_types = [f"{training_method}_{noise_type}_epistemic", 
    f"{training_method}_{noise_type}_aleatoric", 
    f"{training_method}_{noise_type}_both"]
label_names = ["Epistemic (AR1)", "Aleatoric (AR1)", "Both (AR1)"]
save_prefix = f"{training_method}_{noise_type}_"

plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    ylim=2, samples_per_bin=100)


# Set up model and types of simulations to plot
model_name = f"BayesianNN_16_16_N100_priorNormal(0,1.0)"
training_method = "VI" # VI or mcmc_{kernel_name}
noise_type = "WN"

run_types = [f"{training_method}_{noise_type}_epistemic", 
    f"{training_method}_{noise_type}_aleatoric", 
    f"{training_method}_{noise_type}_both"]
label_names = ["Epistemic (indep.)", 
                "Aleatoric (indep)", 
                "Both (indep)"]
save_prefix = f"{training_method}_{noise_type}_"
    

plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    ylim=2, samples_per_bin=100)

noise_type = "AR1"
run_types = [f"{training_method}_{noise_type}_epistemic", 
    f"{training_method}_{noise_type}_aleatoric", 
    f"{training_method}_{noise_type}_both"]
label_names = ["Epistemic (AR1)", "Aleatoric (AR1)", "Both (AR1)"]
save_prefix = f"{training_method}_{noise_type}_"

plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=save_prefix, 
    ylim=2, samples_per_bin=100)