import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pyro 
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_weather
from utils.summary_stats import summary_stats

def plot_uncertainty_with_N_online(params, model_start, N_trains, run_types, label_names, save_prefix="", fname="X_dtf"):
    """Plots ensembles - either shading for 1 std or spaghetti plot of each ensemble member"""
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_paths = [f'{data_path}/{model_start}{N_train}/' for N_train in N_trains]

    # Online
    plot_path = f'{data_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    X_truth = np.load(f"{data_path}/{fname}.npy")

    # Load ml param model results
    filename = f'{model_paths[0]}/{run_types[2]}_{fname}.npy'
    print(filename)
    X = np.load(filename)
    print(X.shape)

    X_mls = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for run_type in run_types] for model_path in model_paths])
    print(X_mls.shape)
    n_ens = X_mls.shape[2]

    # For all plots, time separation assumed to be 10
    T = 10
    nt = int(T/dt_f)
    print(nt, X_mls[0].shape[2])
    N_init = X_mls[0].shape[2] // nt
    # Should be 1
    print(N_init)

    time = np.arange(0, T, dt_f)

    # Calc var across ensembles and average over K
    var = np.var(X_mls, axis=2).mean(axis=-1)
    print(var.shape)

    # At a few different timesteps, plot N_train against percentage variance 
    time_mtus = [0.5, 0.75, 1.0, 1.25, 1.5]
    for t in range(len(time_mtus)):
        time_mtu = time_mtus[t]
        time_ind = int(time_mtu/dt_f)
        var_t = var[..., time_ind]
        
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        for n, N_train in enumerate(N_trains):
            for r in range(len(run_types)):
                ax.bar(N_train*8, var_t[n, r],
                width=40, alpha=0.5, facecolor=plotcolor(run_types[r]), 
                edgecolor=plotcolor(run_types[r]), lw=2,
                label=label_names[r] if n==0 else None)
        ax.axis(xmin=0.)
        ax.legend(loc="upper right")
        ax.set_ylabel(f"Variance")
        ax.set_xlabel("N training data")
        plt.tight_layout()
        plt.savefig(f"{plot_path}{save_prefix}variance_with_N_time_{t}.png")
        print(f"Saved as {plot_path}{save_prefix}variance_with_N_time_{t}.png")
        plt.close()

def plot_uncertainty_with_N_offline(params, model_start, N_trains, save_prefix=""):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_paths = [f'{data_path}/{model_start}{N_train}/' for N_train in N_trains]

    plot_path = f'{data_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Get data it hasnt seen before
    X = np.load(f'{data_path}/X_dtf.npy')
    U = np.load(f'{data_path}/U_dtf.npy')
    print(f'Data loaded from {data_path}')

    # Subsample to remove correlations
    subsample = 1000 # (1 Time Units)
    X = X[::subsample]
    U = U[::subsample]

    N = X.shape[0]

    # Calc variance across validation dataset (same size for all N_train)
    features = np.ravel(X[:])   
    targets = np.ravel(U[:])   
    X_torch = torch.tensor(features, dtype=torch.float32).reshape((-1, 1))
    Y_torch = torch.tensor(targets, dtype=torch.float32).reshape((-1, 1))

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for n, N_train in enumerate(N_trains):
        # Load NN 
        output_dicts = torch.load(f"{model_paths[n]}/model_best.pt")
        model = output_dicts["model"]
        guide = output_dicts["guide"]
        pyro.get_param_store().load( f"{model_paths[n]}/pyro_best_params.pt")
        # Predictive 
        num_samples = 800
        predictive = Predictive(model, guide=guide, num_samples=num_samples,
                                return_sites=("obs", "_RETURN"))
        samples = predictive(X_torch)
        pred_summary = summary_stats(samples)

        # Epistemic only - mean variance across whole dataset
        epistemic_var = (pred_summary["_RETURN"]["std"]**2).mean()
    
        # Both - mean variance across whole dataset
        both_var = (pred_summary["obs"]["std"]**2).mean()

        # Aleatoric only 
        # Deterministic prediction 
        fixed_param_NN = model.get_fixed_param_NN(guide.median())
        fixed_param_NN.eval()
        det_pred = fixed_param_NN(X_torch).detach()
        mean_pred = det_pred[:, 0]
        aleatoric_samples = torch.zeros((num_samples, det_pred.shape[0]))
        for n in range(num_samples):
            aleatoric_samples[n, :] = model.sample_obs(det_pred).detach().squeeze()
        aleatoric_var = (torch.std(aleatoric_samples, dim=0)**2).mean()

        print(epistemic_var, both_var, aleatoric_var)

        
        ax.bar(N_train*8, both_var.detach().numpy(),
                width=40, alpha=0.5, facecolor=plotcolor("both"), 
                edgecolor=plotcolor("both"), lw=2,
                label="Both" if n==0 else None)
        ax.bar(N_train*8, aleatoric_var.detach().numpy(),
                width=40, alpha=0.5, facecolor=plotcolor("aleatoric"), 
                edgecolor=plotcolor("aleatoric"), lw=2,
                label="Aleatoric" if n==0 else None)
        ax.bar(N_train*8, epistemic_var.detach().numpy(),
                width=40, alpha=0.5, facecolor=plotcolor("epistemic"), 
                edgecolor=plotcolor("epistemic"), lw=2,
                label="Epistemic" if n==0 else None)
    ax.axis(xmin=0.)
    ax.legend(loc="upper right")
    ax.set_ylabel(f"Variance")
    ax.set_xlabel("N training data")
    plt.tight_layout()
    plt.savefig(f"{plot_path}{save_prefix}variance_with_N_offline.png")
    print(f"Saved as {plot_path}{save_prefix}variance_with_N_offline.png")
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
    N_trains = [12, 20, 25, 40, 50, 60, 75, 80, 100, 120, 125, 140, 200]

    model_start = f"BayesianNN_16_16_N"
    run_types = [ "both_1",  "aleatoric_1", "epistemic_1"] 
    label_names = [ "Both", "Aleatoric","Epistemic" ]

    plot_uncertainty_with_N_offline(params, model_start, N_trains)
    plot_uncertainty_with_N_online(params, model_start, N_trains, run_types, label_names,  fname="X_dtf")


