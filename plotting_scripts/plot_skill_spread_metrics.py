import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot


from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_weather

def plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=""):
    """Plots spread skill metrics """
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'
    filenames = [f'{model_path}/{run_type}_X_dtf.npy' for run_type in run_types]
    print(filenames)

    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    X_truth = np.load(f"{data_path}/X_dtf.npy")

    # Load ml param model results - must all be same size
    X_mls = np.stack([np.load(filename) for filename in filenames])
    n_ens = X_mls.shape[1]
    # Reshape for initial conditions
    T = 10
    nt = int(T/dt_f)
    N_init = X_mls.shape[2] // nt
    time = np.arange(0, T, dt_f)
    X_mls = X_mls.reshape((len(filenames), n_ens, N_init, nt, K))
    # truth
    N_init_truth = X_truth.shape[0] // nt 
    X_truth = X_truth.reshape(((N_init_truth, nt, K)))[:N_init]

    # Compute mean and var across ensemble (axis=1)
    X_mean = X_mls.mean(axis=1)
    X_var = X_mls.var(axis=1)

    X_diff = (X_mean - X_truth)**2

    print(X_diff.shape, X_mean.shape)

    timesteps = np.linspace(dt_f, 1.5/dt_f, 40)

    for t in timesteps:
        t = int(t)

        n_samples = N_init * K
        print(n_samples)
        samples_per_bin = 25
        n_bins = n_samples // samples_per_bin
        print(n_bins)

        ylim=6.

        plt.clf()
        fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
        
        for r in range(len(run_types)):
            
            # Flatten to combine N_init initial conditions and K dimensions 
            X_diff_rt = X_diff[r, :, t].flatten()
            X_var_rt = X_var[r, :, t].flatten()
            
            # Sort into order of increasing spread
            sorted_inds = np.argsort(X_var_rt.flatten())
            X_diff_sorted = X_diff_rt[sorted_inds]
            X_var_sorted = X_var_rt[sorted_inds]

            # Reshape into bins
            X_diff_bins = X_diff_sorted.reshape((n_bins, samples_per_bin))
            X_var_bins = X_var_sorted.reshape((n_bins, samples_per_bin))

            # Average across variances and take std of err
            spread = np.sqrt(X_var_bins.mean(axis=-1))
            sigma_err = np.sqrt(X_diff_bins.var(axis=-1))

            plt.scatter(spread, sigma_err, 
                color=plotcolor(run_types[r]),
                alpha=0.5,
                label=label_names[r])
        plt.legend(loc="lower right")
        plt.xlabel("r.m.s spread")
        plt.ylabel("r.m.s error")
        plt.title(f"Time = {t*dt_f:.1f}")
        plt.plot([0, ylim], [0, ylim], 'k--')
        plt.axis(xmin=0, xmax=ylim, ymin=0, ymax=ylim)


        plt.savefig(f"{plot_path}/{save_prefix}spread_v_sigma_err_{t:03d}.png")
        print(f"Saved to {plot_path}/{save_prefix}spread_v_sigma_err_{t:03d}.png")    



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
model_name = f"BayesianNN_16_16_N100"
run_types = ["aleatoric", "epistemic"] #, "both_AR1" ] #, "IC_aleatoric", "aleatoric"] #, "aleatoric", "both"] #, "aleatoric_AR1", "both_fix_AR1"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
label_names = ["Aleatoric", "Epistemic"]#, "Both"] #, "Aleatoric+InitCond", "Aleatoric"] #"Epistemic", "Aleatoric", "Both"]
save_prefix = "WN_"

plot_spread_v_skill(params, model_name, run_types, label_names, save_prefix=save_prefix)
