import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_climate

def plot_history_matching(params, model_name, run_types, label_names, 
    save_prefix="", fnames=["X_dtf"], save_step=1):
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
    print(max_pc)
    true_regimes = max_pc//2
    true_regime_wn1 = np.sum(true_regimes==0)
    true_regime_tot = true_regimes.shape[0]*true_regimes.shape[1]
    print(true_regime_wn1 / true_regime_tot)
    print(np.sum(true_regimes==1) / true_regime_tot)
    print(true_regimes.mean())
    print(true_regimes.shape, X_mls.shape)
    true_regimes = true_regimes[:, ::save_step]

    pred_regimes = []
    n_ens = 50
    pred_regimes = np.zeros((len(run_types), N_init, n_ens, len_time))
    for r in range(len(run_types)):
        for i in range(N_init):
            for m in range(n_ens):
                X_transformed = pca.transform(X_mls[r, i, m])
                max_pc = np.argmax(X_transformed, axis=1)
                pred_regimes[r, i, m, :] = max_pc//2

    time_inds = range(100, len_time, 1)
    time = np.arange(100*dt_f, len_time * dt_f * save_step, 1 * dt_f * save_step)[:len(time_inds)]
    print(len(time_inds), time.shape, pred_regimes.shape)

    percent_spent_in_regime_1 = np.zeros((len(run_types), N_init, n_ens, len(time_inds)))
    true_percent_spent_in_regime_1 = np.zeros((N_init, len(time_inds)))
    for j, t in enumerate(time_inds):
        percent_spent_in_regime_1[..., j] = pred_regimes[..., :t].mean(axis=-1)
        true_percent_spent_in_regime_1[:, j] = true_regimes[:, :t].mean(axis=-1)

    # "History matching" over first T=20 MTU - sort by those that are closest to the truth over the first 20MTU
    t_hm_start = int(5 / dt_f)
    t_hm_end = int(25 / dt_f)
    print(percent_spent_in_regime_1.shape)
    errs = np.abs(percent_spent_in_regime_1-true_percent_spent_in_regime_1.mean())
    hm_errs = (errs[..., t_hm_start:t_hm_end]).mean(axis=-1)
    print(hm_errs.shape)
    inds = np.argsort(hm_errs)
    print(inds.shape)
    print(inds)
    sorted_percent_spent_in_regime = percent_spent_in_regime_1[:,:,inds.squeeze(), :]
    print(sorted_percent_spent_in_regime.shape)

    # Take 20% of trajectories that are closest
    percent_match = 0.2
    num_match = int(0.2 * n_ens)
    num_discard = n_ens - num_match
    print(f"History matching - we will keep {num_match} and discard {num_discard}")
    # plot first num_match in the darker colour and remaining "discarded" in lighter colour
    match_color = "darkorchid"
    all_color = "thistle"

    colors = [match_color] * num_match + [all_color] * num_discard

    # Plot individual ensemble members
    fig, ax = plt.subplots(1, figsize=(9, 5))
    ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=1.5, label="Truth")

    # Plot mean and spread
    match_percent_spent_in_regime = sorted_percent_spent_in_regime[:, :, 0:num_match, :]
    discard_percent_spent_in_regime = sorted_percent_spent_in_regime[:, :, num_match:n_ens, :]

    for r in range(len(run_types)):
        for i in range(N_init):
            for m in range(n_ens - num_match):
                ax.plot(time, discard_percent_spent_in_regime[r, i, m], 
                    color = all_color,
                    label = "All PPE members" if m==0 else None,
                    lw=1.5, alpha = 0.7)
            for m in range(num_match):
                ax.plot(time, match_percent_spent_in_regime[r, i, m], 
                    color = match_color,
                    label = "Constrained PPE members" if m==0 else None,
                    lw=1.5, alpha = 0.7)
    plt.legend()
    plt.xlabel("Time (MTU)")
    plt.ylabel("Fraction of time spent in regime $k=1$")
    add_axis_climate(ax, 10, 1)

    plt.savefig(f"{plot_path}/{save_prefix}hm_ens_mem.png")
    print(f"{plot_path}/{save_prefix}hm_ens_mem.png")


    # Mean over N_init and n_ens
    regime_timeseries_mean = sorted_percent_spent_in_regime.mean(axis=(1, 2))
    regime_timeseries_std = sorted_percent_spent_in_regime.std(axis=(1, 2))
    # For those that match only
    hm_regime_timeseries_mean = match_percent_spent_in_regime.mean(axis=(1, 2))
    hm_regime_timeseries_std = match_percent_spent_in_regime.std(axis=(1, 2))

    # For truth just mean over N_init
    true_timeseries_mean = true_percent_spent_in_regime_1.mean(axis=0)
    true_timeseries_std = true_percent_spent_in_regime_1.std(axis=0)
    plt.clf()
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
    for r in range(len(run_types)):
        ax.plot(time, regime_timeseries_mean[r], 
            label="all",
            alpha=0.8, lw=2,
            color=all_color)
            
        # Shading
        ax.fill_between(time, regime_timeseries_mean[r] - regime_timeseries_std[r], 
            regime_timeseries_mean[r] + regime_timeseries_std[r], 
            color = all_color, 
            lw=2, alpha = 0.1)

        # Now add the history matched spread over the top
        ax.plot(time, regime_timeseries_mean[r], 
            label="history matched",
            alpha=0.8, lw=2,
            color=match_color)
            
        # Shading
        ax.fill_between(time, hm_regime_timeseries_mean[r] - hm_regime_timeseries_std[r], 
            hm_regime_timeseries_mean[r] + hm_regime_timeseries_std[r], 
            color = match_color, 
            lw=2, alpha = 0.1)

    plt.legend()
    plt.xlabel("Time (MTU)")
    plt.ylabel("Fraction of time spent in regime $k=1$")
    add_axis_climate(ax)

    plt.savefig(f"{plot_path}/{save_prefix}hm_ens_spread.png")
    print(f"{plot_path}/{save_prefix}hm_ens_spread.png")



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
    run_types = ["epistemic_fix"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic"]
    save_prefix = ""
    fnames = [f"run{i:02d}_X_dtf" for i in range(1)]

    plot_history_matching(params, model_name, run_types, label_names, save_prefix=save_prefix, fnames = fnames)
