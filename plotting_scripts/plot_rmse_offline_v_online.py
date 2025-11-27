import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot
from utils.crps import crps

from plotting_scripts.plot_dicts import plotcolor

def plot_offline_v_online(params, model_name, run_types_offline, run_types_online, 
    label_names, save_prefix="", plot_spread=True):
    """Plots error trajectories """
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'
    filenames_offline = [f'{model_path}/{run_type}_X_dtf.npy' for run_type in run_types_offline]
    filenames_online = [f'{model_path}/{run_type}_X_dtf.npy' for run_type in run_types_online]

    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    X_truth = np.load(f"{data_path}/X_dtf.npy")

    # Load ml param model results
    X_mls_offline = [np.load(filename) for filename in filenames_offline]
    X_mls_online = [np.load(filename) for filename in filenames_online]

    # For all plots, time separation assumed to be 10
    T = 10
    nt = int(T/dt_f)
    N_init_off = X_mls_offline[0].shape[1] // nt
    N_init_on = X_mls_online[0].shape[1] // nt
    N_init = min(N_init_on, N_init_off)
    time = np.arange(0, T, dt_f)

    print(f"{N_init} initial conditions separated by {nt} time units")
    X_init_conds = X_truth[0:N_init:nt]

    # Reshape
    N_init_truth = X_truth.shape[0] // nt 
    X_truth = X_truth.reshape(((N_init_truth, nt, K)))[:N_init]

    # Store errors and std for plotting
    RMSE_off, RMSE_on, STD_off, STD_on = [], [], [], []
    for X_ml_off, X_ml_on in zip(X_mls_offline, X_mls_online):
        # Calc RMSE offline (timeseries)
        X_m_off = X_ml_off.mean(axis=0)
        X_m_off = X_m_off.reshape((N_init_off, nt, K))[:N_init] # Avg across K
        RMSE_off.append(np.sqrt(((X_m_off - X_truth)**2).mean(axis=(2))))

        # Online
        X_m_on = X_ml_on.mean(axis=0)
        X_m_on = X_m_on.reshape((N_init_on, nt, K))[:N_init]
        RMSE_on.append(np.sqrt(((X_m_on - X_truth)**2).mean(axis=(2))))

        # Calc STD offline
        X_var_off = X_ml_off.var(axis=0)
        X_var_off = X_var_off.reshape((N_init_off, nt, K))[:N_init]
        STD_off.append(np.sqrt(X_var_off.mean(axis=(2))))


        X_var_on = X_ml_on.var(axis=0)
        X_var_on = X_var_on.reshape((N_init_on, nt, K))[:N_init]
        STD_on.append(np.sqrt(X_var_on.mean(axis=(2))))
    
    # Plot offline against online over the timeseries
    for t in range(0, nt, 40):
        plt.clf()
        fig, axs = plt.subplots(1, 1, figsize=(6, 4))
        for n in range(len(X_mls_offline)):
            axs.scatter(RMSE_off[n][:, t], RMSE_on[n][:, t], 
                color=plotcolor(run_types_online[n]), 
                label = label_names[n],
                alpha = 0.5)
        #axs.axis(xmin=0, xmax=10)
        axs.legend(loc="lower right")
        axs.set_ylabel(f"Online RMSE")
        axs.set_xlabel("Offline RMSE")
        plt.title(f"Errors at time {time[t]}")
        plt.tight_layout()
        plt.savefig(f"{plot_path}/{save_prefix}offline_v_online_{t}.png")
        print(f"Saved to {plot_path}/{save_prefix}offline_v_online_{t}.png")
        



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
    run_types_online = ["epistemic", "aleatoric", "both"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    run_types_offline = ["offline_epistemic", "offline_aleatoric", "offline_both"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    

    plot_offline_v_online(params, model_name, run_types_offline, run_types_online, label_names,)


