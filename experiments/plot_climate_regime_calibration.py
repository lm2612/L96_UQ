import numpy as np
import matplotlib.pyplot as plt

import torch

from plotting_scripts.plot_regime_uncertainty_after_calibration import plot_regime_uncertainty_distributions_calibration


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


run_types = ["epistemic_fix", "both_fix_AR1"] #,  ["epistemic_fix", "both_fix_AR1", ...]
label_names = [ "Epistemic", "Both"]

 
fnames_true_calibration = [f"climate_F20_run{i:02d}_X_dtf" for i in range(10)]

Fs = [20, 16, 24]
Ts = range(6000, 20001, 500)

Ts = range(6000, 20001, 500)

Fs=[20, 16, 24]
"""
for F in Fs:
    fnames = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(10)]
    save_prefix = f"climate_F{F}_run00-09_"
    for T in Ts:
        for r in range(len(run_types)):
            plot_regime_uncertainty_distributions_calibration(params, model_name, 
                [run_types[r]], fnames_true_calibration, [run_types[r]], [label_names[r]], 
                save_prefix=save_prefix+f"{run_types[r]}", fnames = fnames, save_step = 10,
                T_calib = 6000, T_plot = T, title = f"F={F}")
"""
run_types_other = ["aleatoric_AR1", "epistemic_fix"]
label_names_other = ["Aleatoric", "Epistemic"]
run_types = ["both_fix_AR1"]
label_names = ["Both"]
Fs=[20, 16, 24]

for F in Fs:
    fnames = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(10)]
    save_prefix = f"climate_F{F}_run00-09_"
    for T in Ts:
        plot_regime_uncertainty_distributions_calibration(params, model_name, 
            run_types, fnames_true_calibration, run_types, label_names, 
            run_types_other = run_types_other, label_names_other = label_names_other,
            save_prefix=save_prefix+f"calib_both_compare_all", fnames = fnames, save_step = 10,
            T_calib = 6000, T_plot = T, title = f"F={F}")
