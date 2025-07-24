import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot
from utils.crps import crps

from plotting_scripts.plot_dicts import plotcolor

def plot_error_trajectories(params, model_name, run_types, label_names, save_prefix="", plot_spread=True):
    """Plots error trajectories """
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

    # Load ml param model results
    X_mls = [np.load(filename) for filename in filenames]

    # For all plots, time separation assumed to be 10
    T = 10
    nt = int(T/dt_f)
    N_init = X_mls[0].shape[1] // nt
    time = np.arange(0, T, dt_f)

    print(f"{N_init} initial conditions separated by {nt} time units")
    X_init_conds = X_truth[0:N_init:nt]

    # Reshape
    N_init_truth = X_truth.shape[0] // nt 
    X_truth = X_truth.reshape(((N_init_truth, nt, K)))[:N_init]

    # Plot error trajectories
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for X_ml, run_type, label_name in zip(X_mls, run_types, label_names):
        # Take mean prediction across ensembles (only relevant if n_ens > 1)
        X_m = X_ml.mean(axis=0)
        #N_init = X_ml.shape[0] // nt
        X_m = X_m.reshape((N_init, nt, K))
        # Take rmse across all initial conditions (axis=0) and variables (axis=2)
        X_diff = np.sqrt(((X_m - X_truth)**2).mean(axis=(0, 2)))
        axs.plot(time[0:nt], X_diff, 
            label=label_name, 
            alpha=0.8,
            color=plotcolor(run_type))

        if plot_spread:
            # Get spread in err across ensembles
            X_var = X_ml.var(axis=0)
            X_var = X_var.reshape((N_init, nt, K))
            # Take mean across all initial conditions (axis=0) and variables (axis=2)
            X_std = np.sqrt(X_var.mean(axis=(0, 2)))

            axs.plot(time[0:nt], X_std, 
                alpha=0.8,
                color=plotcolor(run_type),
                linestyle="dashed")

    axs.axis(xmin=0, xmax=10)
    axs.legend(loc="lower right")
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/X_rmse_timeseries.png")
    print(f"Saved to {plot_path}/X_rmse_timeseries.png")

    # Plot CRPS
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)

    for X_ml, run_type, label_name in zip(X_mls, run_types, label_names):
        n_ens = X_ml.shape[0]
        if n_ens > 1:
            X_ml = X_ml.reshape((n_ens, N_init, nt, K))
            err = crps(X_truth, X_ml).mean(axis=-1)
            axs.plot(time[0:nt], err, 
                label=label_name, 
                alpha=0.8,
                color=plotcolor(run_type))
    axs.axis(xmin=0, xmax=4)
    axs.legend(loc="lower right")
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    plt.tight_layout()
    plt.savefig(f"{plot_path}/X_crps_timeseries.png")

    print(f"Saved to {plot_path}/X_crps_timeseries.png")