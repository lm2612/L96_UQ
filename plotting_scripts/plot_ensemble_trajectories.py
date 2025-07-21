import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plotting_scripts.plot_dicts import plotcolor

def plot_ensembles(params, model_name, run_types, label_names, save_prefix="", 
        shading=True, spaghetti=False):
    """Plots ensembles - either shading for 1 std or spaghetti plot of each ensemble member"""
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

    for i in range(N_init):
        print(f"Plotting initial condition {i}")
        
        plt.clf()
        fig, axs = plt.subplots(4, 2, figsize=(14, 12)) 
        axs = axs.flatten()
        for k in range(K):
            axs[k].plot(time[0:nt], X_truth[i*nt:(i+1)*nt, k],
            label="Truth", 
            alpha=1.,
            color=plotcolor("Truth"))
            for X_ml, run_type, label_name in zip(X_mls, run_types, label_names):
                n_ens = X_ml.shape[0]
                if shading:
                    mean_X = X_ml[:, i*nt:(i+1)*nt, k].mean(axis=0)
                    std_dev = X_ml[:, i*nt:(i+1)*nt, k].std(axis=0)
                    axs[k].fill_between(time[0:nt], 
                        mean_X - std_dev, 
                        mean_X + std_dev ,
                        alpha=0.2,
                        color=plotcolor(run_type))
                if spaghetti:
                    for n in range(n_ens):
                        axs[k].plot(time[0:nt], X_ml[n, i*nt:(i+1)*nt, k],
                        alpha=0.4,
                        color=plotcolor(run_type))
                # Plot mean (regardless of type of plot)
                axs[k].plot(time[0:nt], X_ml[:, i*nt:(i+1)*nt, k].mean(axis=0),
                    label=label_name, 
                    alpha=0.8, lw=2,
                    color=plotcolor(run_type))
                
            axs[k].axis(xmin=0, xmax=2., ymin = -10., ymax=16.)
            axs[k].legend(loc="upper left")
            axs[k].set_ylabel(f"X_{k}")
            axs[k].set_xlabel("Time")
            axs[k].set_title(f"Initial Condition {i}")
            axs[k].legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{plot_path}{save_prefix}ensemble_timeseries_{i}.png")
        print(f"Saved as {plot_path}{save_prefix}ensemble_timeseries_{i}.png")
        plt.close()

if __name__ == "__main__":
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
    N_train = 50
    model_name = f"BayesianNN_16_N{N_train}"
    run_types = ["epistemic", "aleatoric", "both"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    save_prefix = "whitenoise_"

    plot_ensembles(params, model_name, run_types, label_names, save_prefix=save_prefix)


