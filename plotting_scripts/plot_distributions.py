import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plotting_scripts.plot_dicts import plotcolor


def plot_distributions(params, model_name, run_types, label_names, fnames="X_dtf",
    fnames_true=None, det=False, save_prefix="", linestyles = None, spatial_mean=False):
    """Plots distributions """
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    if isinstance(fnames, str):
        fnames = [fnames]
    if fnames_true is None:
        fnames_true = fnames
    elif isinstance(fnames_true, str):
        fnames_true = [fnames_true]
    

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'


    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    print([f'{data_path}/{fname}.npy' for fname in fnames_true])
    X_truth = np.stack([np.load(f'{data_path}/{fname}.npy') for fname in fnames_true])
    print(X_truth.shape)
    if spatial_mean:
        X_truth = X_truth.mean(axis=-1)
        print(X_truth.shape)

    # Load ml param model results - must all be same size
    print([[(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames] for run_type in run_types])
    X_mls = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames] for run_type in run_types])
    print(X_mls.shape)
    if spatial_mean:
        X_mls = X_mls.mean(axis=-1)
        print(X_mls.shape)

    # Load det if its provided
    if det:
        X_det = np.stack([np.load(f'{model_path}/deterministic_{fname}.npy') for fname in fnames])
        if spatial_mean:
            X_det = X_det.mean(axis=-1)
            print(X_det.shape)

    # Plot distributions
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True)
    plt.sca(axs[0])
    X_domain = np.linspace(-15., 20., 100)

    pdf_truth = kde_plot(X_truth[:], X_domain)
    axs[0].plot(X_domain, pdf_truth, color="black", label="Truth")

    if det:
        pdf_det = kde_plot(X_det[:], X_domain)
        axs[0].plot(X_domain, pdf_det, color="lightblue", label="Deterministic", alpha=0.8)
        axs[1].plot(X_domain, pdf_det-pdf_truth, color="lightblue", alpha=0.8)

    for r in range(len(run_types)):
        pdf = kde_plot(X_mls[r], X_domain)
        axs[0].plot(X_domain, pdf, 
                label=label_names[r],
                color=plotcolor(run_types[r]),
                alpha=0.8, 
                linestyle=linestyles[r] if linestyles is not None else "solid")

        axs[1].plot(X_domain, pdf-pdf_truth, 
                #label=label_names[r],
                color=plotcolor(run_types[r]),
                alpha=0.8, 
                linestyle=linestyles[r] if linestyles is not None else "solid")

    
    plt.sca(axs[0])
    plt.xlabel("$X$")
    plt.ylabel("p.d.f ")

    plt.legend(loc="upper left")

    # Plot diff in distributions from truth
    plt.sca(axs[1])
    plt.axhline(0., color="black", linestyle="dotted")
    
    plt.xlabel("$X$")
    plt.ylabel("p.d.f - True p.d.f")

    #plt.legend()
    plt.tight_layout()
    if spatial_mean:
        ext = "_spatial_mean"
    else:
        ext = ""
    plt.savefig(f"{plot_path}{save_prefix}X_pdf_and_diff{ext}.png")
    print(f"Saved to {plot_path}{save_prefix}X_pdf_and_diff{ext}.png")


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
    run_types = ["epistemic", "aleatoric", "both"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    save_prefix = ""

    #plot_distributions(params, model_name, run_types, label_names, save_prefix=save_prefix)


    N_train = 100
    F_test = 20
    model_name = f"BayesianNN_16_16_N{N_train}"
    run_types = ["epistemic_fix", "aleatoric_AR1", "both_fix_AR1", "epistemic_fix"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic (fix)", "Aleatoric (AR1)", "Both", "Epistemic (PPE)"]
    linestyles = ["solid"]*3 + ["dashed"]*3
    save_prefix = f"climate_F{F_test}_"
    fnames = [f"climate_F{F_test}_run{i:02d}_X_dtf" for i in range(1)]
    fnames_true = [f"climate_F{F_test}_run{i:02d}_X_dtf" for i in range(10)]


    #plot_regime_uncertainty_time(params, model_name, run_types, label_names, 
    #    save_prefix=save_prefix, fnames = fnames, save_step=10) 

    plot_distributions(params, model_name, run_types, label_names, save_prefix=save_prefix, 
        fnames=fnames, fnames_true=fnames, linestyles=linestyles) 


