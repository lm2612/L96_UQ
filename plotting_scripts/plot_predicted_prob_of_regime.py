import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_weather

def plot_prob_of_regime(params, model_name, run_types, label_names, save_prefix="", fname="X_dtf"):
    """For s2s predictions"""
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

    # Average over ensemble to get the predicted probability of regime 1 with time
    mean_regime = pred_regimes.mean(axis=1)
    spread_regime = pred_regimes.std(axis=1)

    # Plot against true regime as a trajectory
    max_time = 10
    time = np.arange(0., max_time, dt_f)
    mean_regime = mean_regime[:, :len(time)]
    spread_regime = spread_regime[:, :len(time)]
    true_regime = true_regimes[:len(time)]
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(14, 12)) 
    ax.plot(time, true_regime,
        label="Truth", 
        alpha=1.,
        color=plotcolor("Truth"))
    for r in range(len(run_types)):
        ax.fill_between(time, mean_regime[r] - spread_regime[r],
            mean_regime[r] + spread_regime[r],
            alpha=0.2,
            color=plotcolor(run_types[r]))
        ax.plot(time, mean_regime[r],
            label=label_names[r],
            alpha=0.8, lw=2,
            color=plotcolor(run_types[r]))
                
    ax.axis(xmin=0, xmax=3., ymin = -0.5, ymax=1.5)
    ax.set_yticks([0., 1.])
    ax.legend(loc="upper left")
    ax.set_ylabel(f"Regime")
    ax.set_xlabel("Time")
    add_axis_weather(ax, max_days=16, step_days=5)
    plt.tight_layout()
    plt.savefig(f"{plot_path}{save_prefix}predicted_probability_of_regime.png")
    print(f"Saved as {plot_path}{save_prefix}predicted_probability_of_regime.png")
    plt.close()

    # Plot spread/skill 
    plt.clf()
    fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
    for r in range(len(run_types)):
        plt.scatter(mean_regime[r, :100], spread_regime[r, :100],
            color=plotcolor(run_types[r]),
            alpha=0.5,
            label=label_names[r])
    plt.legend(loc="lower right")
    plt.xlabel("mean regime")
    plt.ylabel("std in regime")
    #plt.plot([0, ylim], [0, ylim], 'k--')
    #plt.axis(xmin=0, xmax=ylim, ymin=0, ymax=ylim)

    plt.savefig(f"{plot_path}/{save_prefix}predicted_prob_v_prob_of_occ.png")
    print(f"Saved to {plot_path}/{save_prefix}predicted_prob_v_prob_of_occ.png")    



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
    run_types = ["epistemic_fix"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    save_prefix = "whitenoise_"
    fname = "run00_X_dtf"

    plot_prob_of_regime(params, model_name, run_types, label_names, save_prefix=save_prefix, fname = fname)
