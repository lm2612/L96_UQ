import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor


def plot_regime_uncertainty_time(params, model_name, run_types, label_names, save_prefix="", fname="X_dtf"):
    """Plots ensembles - either shading for 1 std or spaghetti plot of each ensemble member"""
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
    
    # Load ml param model results
    X_mls = [np.load(filename) for filename in filenames]

    # Load PCA object
    pca = np.load(f"{data_path}/pca_fit.npy", allow_pickle=True).item()
    print(pca)

    ## How often is our simulation in each 'regime' over the entire timeseries - look for dominant PCs
    X_transformed = pca.transform(X_truth)
    max_pc = np.argmax(X_transformed, axis=1)
    true_regimes = max_pc//2
    true_regime_wn1 = np.sum(true_regimes==0)
    true_regime_tot = true_regimes.shape[0]
    print(true_regimes.shape)
    print(true_regime_wn1 / true_regime_tot)

    len_time = X_mls[0].shape[1]


    ### I AM HERE
    pred_regimes = []
    n_ens = 50
    pred_regimes = np.zeros((len(run_types), n_ens, len_time))
    for r in range(len(run_types)):
        X_ml = X_mls[r]
        n_ens  = X_ml.shape[0]
        print(X_ml.shape)
        for m in range(n_ens):
            X_transformed = pca.transform(X_ml[m])
            max_pc = np.argmax(X_transformed, axis=1)
            pred_regimes[r, m, :] = max_pc//2

    time_inds = range(100, len_time, 100)
    time = np.arange(100*dt_f, len_time * dt_f, 100 * dt_f)
    percent_spent_in_regime_1 = np.zeros((len(run_types), n_ens, len(time_inds)))
    for j, t in enumerate(time_inds):
        percent_spent_in_regime_1[:, :, j] = pred_regimes[:, :, :t].mean(axis=-1)


    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
    for r in range(len(run_types)):
        mean_percent_spent_in_regime_1 = percent_spent_in_regime_1[r, :].mean(axis=0)
        ax.plot(time, mean_percent_spent_in_regime_1, 
            label=label_names[r],
            alpha=0.8, lw=2,
            color=plotcolor(run_types[r]))
            
        # Shading
        std_percent_spent_in_regime_1 = percent_spent_in_regime_1[r, :].std(axis=0)
        ax.fill_between(time, mean_percent_spent_in_regime_1 - std_percent_spent_in_regime_1, 
        mean_percent_spent_in_regime_1 + std_percent_spent_in_regime_1, 
            color = plotcolor(run_types[r]), 
            lw=2, alpha = 0.1)
    plt.legend()
    plt.xlabel("Time (MTU)")
    plt.ylabel("Fraction of time spent in regime 1")
    plt.savefig(f"{plot_path}/{save_prefix}regime_ens_spread.png")
    print(f"{plot_path}/{save_prefix}regime_ens_spread.png")

    ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
    for r in range(len(run_types)):
        for n in range(n_ens):
            ax.plot(time, percent_spent_in_regime_1[r, n], 
            color = plotcolor(run_types[r]), 
            lw=1, alpha = 0.3)

    plt.savefig(f"{plot_path}/{save_prefix}regime_ens_mem.png")
    print(f"{plot_path}/{save_prefix}regime_ens_mem.png")


    # Get mean and std 
    time_inds = range(10, len_time, 10)
    time = np.arange(10*dt_f, len_time *dt_f, 10 *dt_f)
    regime_timeseries_mean = np.zeros((len(run_types), len(time_inds)))
    regime_timeseries_std = np.zeros((len(run_types), len(time_inds)))
    for r in range(len(run_types)):
        for j, t in enumerate(time_inds):
            mean_ens = pred_regimes[r, :, :t].mean(axis=-1)
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
    model_name = f"BayesianNN_16_N{N_train}"
    run_types = ["epistemic", "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    save_prefix = "whitenoise_"

    plot_regime_uncertainty_time(params, model_name, run_types, label_names, save_prefix=save_prefix)
