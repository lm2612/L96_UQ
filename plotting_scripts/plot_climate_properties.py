import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plotting_scripts.plot_dicts import plotcolor


def plot_climate_properties(params, model_name, run_types, label_names, fname="X_dtf",
    save_prefix="", linestyles = None, quantiles = [0.01, 0.05, 0.25, 0.5, 0.76, 0.95, 0.99], save_step=1):
    """Plots distributions """
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'
    filenames = [f'{model_path}/{run_type}_{fname}.npy' for run_type in run_types]
    print(filenames)

    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    X_truth = np.load(f"{data_path}/{fname}.npy")
    print(X_truth.shape)
    true_quantiles = np.quantile(X_truth, quantiles)
    print(true_quantiles)
 
    # Load ml param model results
    X_mls = np.stack([np.load(filename) for filename in filenames])
    print(X_mls.shape)
    pred_quantiles = np.quantile(X_mls, quantiles, axis=(1,2,3))
    print(pred_quantiles)

    len_quantiles = len(quantiles)
    len_time = X_mls.shape[2]


    time_inds = range(100, len_time, 100)
    time = np.arange(100*dt_f, len_time * dt_f * save_step, 100 * dt_f * save_step)[:len(time_inds)]
    print(len(time_inds))

    pred_quantiles = np.zeros((len(run_types), len(time_inds), len_quantiles))
    
    for j, t in enumerate(time_inds):
        pred_quantiles[:, j, :] = np.quantile(X_mls[:, :, :t, :], quantiles, axis=(1,2,3)).T

    print(pred_quantiles.shape, time.shape)

    # Plot all on same plot
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(10, 6))
    for q in range(len_quantiles):
        ax.axhline(true_quantiles[q], color="black", linestyle="dashed", lw=2, label="Truth" if q == 0 else None)
        ax.text(time[-1], true_quantiles[q], f"q={quantiles[q]}", ha="center", va="bottom" if quantiles[q]>=0.5 else "top")
        for r in range(len(run_types)):
            ax.plot(time, pred_quantiles[r, :, q], 
                label=label_names[r] if q == 0 else None,
                alpha=0.8, lw=2,
                color=plotcolor(run_types[r]))
    plt.legend()
    plt.xlabel("Time (MTU)")
    plt.ylabel(f"Quantile")
    #add_axis_climate(ax)

    plt.savefig(f"{plot_path}{save_prefix}climate_quantiles_all.png")
    print(f"Saved to {plot_path}{save_prefix}climate_quantiles_all.png")
    
    # Individual plots
    for q in range(len_quantiles):
        plt.clf()
        # Plot distributions
        fig, ax = plt.subplots(1, figsize=(10, 6))
        ax.axhline(true_quantiles[q], color="black", linestyle="dashed", lw=2, label="Truth")
        for r in range(len(run_types)):
            ax.plot(time, pred_quantiles[r, :, q], 
                label=label_names[r],
                alpha=0.8, lw=2,
                color=plotcolor(run_types[r]))
        plt.legend()
        plt.xlabel("Time (MTU)")
        plt.ylabel(f"Quantile {quantiles[q]}")
        #add_axis_climate(ax)

        plt.savefig(f"{plot_path}{save_prefix}climate_quantile_{quantiles[q]}.png")
        print(f"Saved to {plot_path}{save_prefix}climate_quantile_{quantiles[q]}.png")

    
    

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
    N_train = 100
    model_name = f"BayesianNN_16_16_N{N_train}"
    run_types = ["epistemic_fix", "aleatoric_AR1", "both_fix_AR1"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    save_prefix = ""

    plot_climate_properties(params, model_name, run_types, label_names, save_prefix=save_prefix,
    fname = "run00_X_dtf"
    )


