import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot
from utils.crps import crps

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_weather

def plot_error_trajectories(params, model_name, run_types, label_names, save_prefix="", 
    plot_spread=True, include_sum=False, linestyles=None):
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

    # Take mean variance across all initial conditions  (axis=1) and dims K (axis=3)
    print(X_mls.shape, X_mean.shape)
    X_var_m = X_var.mean(axis=(1,3))

    # Calc RMSE
    X_rmse = np.sqrt((X_mean - X_truth)**2).mean(axis=(1,3))
    print(X_var_m.shape, X_rmse.shape)

    # Plot error trajectories
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))
    for r in range(len(run_types)):
        axs.plot(time[0:nt], X_rmse[r], 
            label=label_names[r], 
            alpha=0.8,
            color=plotcolor(run_types[r]))

        if plot_spread:
            axs.plot(time[0:nt], np.sqrt(X_var_m[r]), 
                alpha=0.8,
                color=plotcolor(run_types[r]),
                linestyle="dashed")

    axs.axis(xmin=0, xmax=6)
    axs.legend(loc="lower right", ncol=2 if len(run_types)>4 else 1)
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    add_axis_weather(axs,  max_days = 35., step_days = 10.)

    plt.tight_layout()
    plt.savefig(f"{plot_path}/{save_prefix}X_rmse_timeseries.png")
    print(f"Saved to {plot_path}/{save_prefix}X_rmse_timeseries.png")


    # Plot variance only
    variance_sum = 0
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    for r in range(len(run_types)):
        axs.plot(time[0:nt], np.sqrt(X_var_m[r]), 
            alpha=0.8,
            color=plotcolor(run_types[r]),
            label=label_names[r],
            linestyle=linestyles[r] if linestyles is not None else "solid")
        if include_sum and (("epistemic" in run_types[r]) or ("aleatoric")in run_types[r]):
            variance_sum += X_var_m[r]
        
    # Include sum of epistemic and aleatoric
    if include_sum:
        axs.plot(time[0:nt], np.sqrt(variance_sum), 
                alpha=0.8,
                color="gray",
                label="sum",
                linestyle="dashed")

    axs.axis(xmin=0, xmax=6)
    axs.legend(loc="lower right" , ncol=2 if len(run_types)>4 else 1)
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    add_axis_weather(axs,  max_days = 35., step_days = 10.)

    plt.tight_layout()
    plt.savefig(f"{plot_path}/{save_prefix}X_std_timeseries.png")
    print(f"Saved to {plot_path}/{save_prefix}X_std_timeseries.png")

    # Plot CRPS
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)

    for r in range(len(run_types)): 
        X_ml = X_mls[r]
        n_ens = X_ml.shape[0]
        if n_ens > 1:
            X_ml = X_ml.reshape((n_ens, N_init, nt, K))
            err = crps(X_truth, X_ml).mean(axis=-1)
            axs.plot(time[0:nt], err, 
                label=label_names[r], 
                alpha=0.8,
                color=plotcolor(run_types[r]),
                linestyle=linestyles[r] if linestyles is not None else "solid")
    axs.axis(xmin=0, xmax=6)
    axs.legend(loc="lower right", ncol=2 if len(run_types)>4 else 1)
    axs.set_ylabel(f"X")
    axs.set_xlabel("Time")
    add_axis_weather(axs,  max_days = 35., step_days = 10.)
    plt.tight_layout()
    plt.savefig(f"{plot_path}{save_prefix}X_crps_timeseries.png")

    print(f"Saved to {plot_path}/{save_prefix}X_crps_timeseries.png")
