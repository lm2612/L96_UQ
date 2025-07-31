import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_climate

def plot_regime_uncertainty_time(params, model_name, run_types, label_names, save_prefix="", fnames=["X_dtf"]):
    """Climate predictions"""
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    if isinstance(fnames, str):
        fnames = [fnames]

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'

    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    X_truth = np.stack([np.load(f'{data_path}/{fname}.npy') for fname in fnames])
    print(X_truth.shape)
    
    # Load ml param model results - must all be same size
    X_mls = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames] for run_type in run_types])
    print(X_mls.shape)
    n_ens = X_mls.shape[2]
    N_init = X_mls.shape[1]
    len_time = X_mls.shape[3]

    # Load PCA object
    pca = np.load(f"{data_path}/pca_fit.npy", allow_pickle=True).item()
    print(pca)

    ## How often is our simulation in each 'regime' over the entire timeseries - look for dominant PCs
    X_transformed = np.stack([pca.transform(X_truth[i]) for i in range(N_init)])
    max_pc = np.argmax(X_transformed, axis=2)
    true_regimes = max_pc//2
    true_regime_wn1 = np.sum(true_regimes==0)
    true_regime_tot = true_regimes.shape[0]*true_regimes.shape[1]
    print(true_regime_wn1 / true_regime_tot)

    pred_regimes = []
    n_ens = 50
    pred_regimes = np.zeros((len(run_types), N_init, n_ens, len_time))
    for r in range(len(run_types)):
        for i in range(N_init):
            for m in range(n_ens):
                X_transformed = pca.transform(X_mls[r, i, m])
                max_pc = np.argmax(X_transformed, axis=1)
                pred_regimes[r, i, m, :] = max_pc//2

    time_inds = range(100, len_time, 100)
    time = np.arange(100*dt_f, len_time * dt_f, 100 * dt_f)
    percent_spent_in_regime_1 = np.zeros((len(run_types), N_init, n_ens, len(time_inds)))
    for j, t in enumerate(time_inds):
        percent_spent_in_regime_1[..., j] = pred_regimes[..., :t].mean(axis=-1)


    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
    for r in range(len(run_types)):
        mean_percent_spent_in_regime_1 = percent_spent_in_regime_1[r, :].mean(axis=(0,1))
        ax.plot(time, mean_percent_spent_in_regime_1, 
            label=label_names[r],
            alpha=0.8, lw=2,
            color=plotcolor(run_types[r]))
            
        # Shading
        std_percent_spent_in_regime_1 = percent_spent_in_regime_1[r, :].std(axis=(0,1))
        ax.fill_between(time, mean_percent_spent_in_regime_1 - std_percent_spent_in_regime_1, 
        mean_percent_spent_in_regime_1 + std_percent_spent_in_regime_1, 
            color = plotcolor(run_types[r]), 
            lw=2, alpha = 0.1)
    plt.legend()
    plt.xlabel("Time (MTU)")
    plt.ylabel("Fraction of time spent in regime 1")
    add_axis_climate(ax)

    plt.savefig(f"{plot_path}/{save_prefix}regime_ens_spread.png")
    print(f"{plot_path}/{save_prefix}regime_ens_spread.png")

    ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
    for r in range(len(run_types)):
        for i in range(N_init):
            for m in range(n_ens):
                ax.plot(time, percent_spent_in_regime_1[r, i, m], 
                color = plotcolor(run_types[r]), 
                lw=1, alpha = 0.3)

    add_axis_climate(ax)

    plt.savefig(f"{plot_path}/{save_prefix}regime_ens_mem.png")
    print(f"{plot_path}/{save_prefix}regime_ens_mem.png")


    # Get mean and std 
    print(pred_regimes.shape)
    time_inds = range(10, len_time, 100)
    time = np.arange(10*dt_f, len_time*dt_f, 100*dt_f)
    regime_timeseries_mean = np.zeros((len(run_types), len(time_inds)))
    regime_timeseries_std = np.zeros((len(run_types), len(time_inds)))
    for r in range(len(run_types)):
        for j, t in enumerate(time_inds):
            mean_ens = pred_regimes[r, :, :, :t].mean(axis=-1)
            regime_timeseries_mean[r, j] = mean_ens.mean()
            regime_timeseries_std[r, j] =  mean_ens.std()

    ## Plot
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
    for r in range(len(run_types)):
        ax.plot(time, regime_timeseries_mean[r], 
            label=label_names[r],
            alpha=0.8, lw=2,
            color=plotcolor(run_types[r]))
    plt.legend()
    plt.xlabel("Time (MTU)")
    plt.ylabel("Fraction of time spent in regime 1")
    add_axis_climate(ax)

    plt.savefig(f"{plot_path}/{save_prefix}regime_convergence.png")
    print(f"{plot_path}/{save_prefix}regime_convergence.png")

    plt.clf()
    ## Plot
    fig, ax = plt.subplots(1, figsize=(10, 6))
    for r in range(len(run_types)):
        ax.plot(time, regime_timeseries_std[r], 
            label=label_names[r],
            alpha=0.8, lw=2,
            color=plotcolor(run_types[r]))
    #ax.axhline(true_regimes.mean(), color="black", linestyle="dashed")
    plt.legend()
    plt.xlabel("Time (MTU)")
    plt.ylabel("Standard deviation across ensemble in fraction of time spent in regime 1")
    add_axis_climate(ax)
    plt.savefig(f"{plot_path}/{save_prefix}regime_uncertainty_time.png")
    print(f"{plot_path}/{save_prefix}regime_uncertainty_time.png")

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
    model_name = f"BayesianNN_16_16_N{N_train}"
    run_types = ["epistemic_fix", "aleatoric_AR1", "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    save_prefix = "whitenoise_"
    fnames = [f"run{i:02d}_X_dtf" for i in range(5)]

    plot_regime_uncertainty_time(params, model_name, run_types, label_names, save_prefix=save_prefix, fnames = fnames)
