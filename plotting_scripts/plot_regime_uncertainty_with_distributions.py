import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_climate

def plot_regime_uncertainty_distributions(params, model_name, run_types, label_names, 
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
    print(len(time_inds), time.shape, pred_regimes.shape, true_regimes.shape)

    # Assess data up to time T - optional, can be max (-1) or shorter timescales
    T = -1
    pred_regimes = pred_regimes[...,:T]


    # Fracion of time spent in regime 1 for each ensemble member / init cond
    n_init = pred_regimes.shape[1]
    n_ens = pred_regimes.shape[2]
    print(n_init)
    fig, ax = plt.subplots(1, figsize=(10, 6))

    ax.axvline(true_regimes.mean(), color="k")
    for r in range(len(run_types)):
        frac_time = np.zeros((n_init*n_ens))
        for i in range(n_init):
            for m in range(n_ens):
                # For each ensemble member, get proportion of time spent in regime
                frac_time[int(i*n_ens + m)] = pred_regimes[r, i, m].mean(axis=-1)
        print(frac_time)
        ax.hist(frac_time, 
            bins = np.arange(0.2, 0.8, 0.01),
            alpha=0.5, lw=1,
            label=label_names[r],
            color=plotcolor(run_types[r]),
            density=True)
        mean_r = frac_time.mean()
        std_r = frac_time.std()
        ax.axvline(mean_r, color=plotcolor(run_types[r]))
        #ax.axvline(mean_r-std_r, color=plotcolor(run_types[r]))
        #ax.axvline(mean_r+std_r, color=plotcolor(run_types[r]))
    if n_init > 1:
        frac_time = true_regimes.mean(axis=-1)
        ax.hist(frac_time, 
            bins = np.arange(0.2, 0.8, 0.01),
            alpha=.6, lw=1,
            label="Truth",
            color="black", 
            #histtype="step",
             density=True)

    plt.legend()
    plt.xlabel("Fraction of time spent in regime $k=1$")

    plt.savefig(f"{plot_path}/{save_prefix}_frac_time_in_reg_hist.png")
    print(f"{plot_path}/{save_prefix}_frac_time_in_reg_hist.png")




    

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
    model_name = f"BayesianNN_Heteroscedastic_16_16_N100"

    run_types = ["epistemic_fix", "aleatoric_AR1", "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    
    F=20
    fnames = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(7)]
    save_prefix = f"climate_F{F}_run00-06_"

    plot_regime_uncertainty_distributions(params, model_name, run_types, label_names, save_prefix=save_prefix, fnames = fnames)
