import numpy as np
import matplotlib.pyplot as plt

import torch

from plotting_scripts.plot_regime_uncertainty_with_distributions import plot_regime_uncertainty_distributions


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

    run_types = ["aleatoric_AR1",  "epistemic_fix", "both_fix_AR1"] 
    label_names = [ "Aleatoric (AR1)",  "Epistemic (PPE)", "Both"]
    
    Fs = [20] #, 16, 24]
    Ts = range(1000, 20000, 500)

    for F in Fs:
        fnames = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(10)]
        save_prefix = f"climate_F{F}_run00-09_"

        for T in Ts:
            plot_regime_uncertainty_distributions(params, model_name, run_types, label_names, 
                save_prefix=save_prefix+"hist_", fnames = fnames, kde=False, save_step = 10,
                T=T,
                title = f"F={F}")

            plot_regime_uncertainty_distributions(params, model_name, run_types, label_names, 
                save_prefix=save_prefix+"kde_", fnames = fnames, kde=True,  save_step = 10,
                T=T,
                title = f"F={F}")
