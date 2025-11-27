import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plotting_scripts.plot_dicts import plotcolor
from scipy.ndimage import uniform_filter1d


def running_mean_axis(a, W, axis=-1):
    # a : Input ndarray
    # W : Window size
    # axis : Axis along which we will apply rolling/sliding mean
    hW = W//2
    L = a.shape[axis]-W+1   
    indexer = [slice(None) for _ in range(a.ndim)]
    indexer[axis] = slice(hW,hW+L)
    return uniform_filter1d(a,W,axis=axis)[tuple(indexer)]

#def running_mean(X, running_mean_window):
#    return np.convolve(X, np.ones(running_mean_window)/running_mean_window, mode='valid')

#def running_mean_axis(X, running_mean_window, axis=0):
#    return np.apply_along_axis(running_mean(X, running_mean_window), axis=axis, arr=X)


def plot_anomaly_distributions(params, model_name, run_types, label_names, fnames="X_dtf", fnames_base="X_dtf",
    fnames_true=None, det=False, save_prefix="", linestyles = None, running_mean=False, running_mean_window=100):
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
    if running_mean:
        X_truth = X_truth[:, ::10, :]
        X_truth =  running_mean_axis(X_truth, running_mean_window, axis=1) 
        print(X_truth.shape)
    X_truth_base = np.stack([np.load(f'{data_path}/{fname}.npy') for fname in fnames_base])
    print(X_truth_base.shape)
    if running_mean:
        X_truth_base = X_truth_base[:, ::10, :]
        X_truth_base =  running_mean_axis(X_truth_base, running_mean_window, axis=1) 
        print(X_truth_base.shape)
    
    #X_truth = X_truth - X_truth_base
    
    # Load ml param model results - must all be same size
    print([[(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames] for run_type in run_types])
    
    X_mls = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames] for run_type in run_types])
    print(X_mls.shape)

    if running_mean:
        X_mls =  running_mean_axis(X_mls, running_mean_window, axis=3) 
        print(X_mls.shape)

    # Load base model - F=20 only
    print([[(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames_base] for run_type in run_types])
    X_base = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames_base] for run_type in run_types])
    print(X_base.shape)
    if running_mean:
        X_base =  running_mean_axis(X_base, running_mean_window, axis=3) 
        print(X_base.shape)

    #X_mls = X_mls - X_base

    

    # Load det if its provided
    if det:
        X_det = np.stack([np.load(f'{model_path}/deterministic_{fname}.npy') for fname in fnames])
        print(X_det.shape)
        
        X_det_base = np.stack([np.load(f'{model_path}/deterministic_{fname}.npy') for fname in fnames_base])

        if running_mean:
            X_det = running_mean_axis(X_det, running_mean_window, axis=2) 
            X_det_base = running_mean_axis(X_det_base, running_mean_window, axis=2) 
            print(X_det.shape)


    # Plot distributions
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), dpi=100, sharex=True)
    plt.sca(axs[0])
    X_domain = np.linspace(-15., 20., 100)

    pdf_truth = kde_plot(X_truth[:], X_domain)
    pdf_truth_base = kde_plot(X_truth_base[:], X_domain)
    pdf_truth = pdf_truth - pdf_truth_base
    axs[0].plot(X_domain, pdf_truth, color="black", label="Truth")

    if det:
        pdf_det = kde_plot(X_det[:], X_domain)
        pdf_det_base = kde_plot(X_det_base[:], X_domain)
        pdf_det = pdf_det - pdf_det_base
        axs[0].plot(X_domain, pdf_det, color="lightblue", label="Deterministic", alpha=0.8)
        axs[1].plot(X_domain, pdf_det-pdf_truth, color="lightblue", alpha=0.8)

    for r in range(len(run_types)):
        pdf = kde_plot(X_mls[r], X_domain)
        pdf_base = kde_plot(X_base[r], X_domain)
        pdf = pdf - pdf_base
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
    if running_mean:
        ext = f"_running_mean{running_mean_window}"
    else:
        ext = ""
    plt.savefig(f"{plot_path}{save_prefix}X_anom_pdf_and_diff{ext}.png")
    print(f"Saved to {plot_path}{save_prefix}X_anom_pdf_and_diff{ext}.png")


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
    model_name = f"BayesianNN_Heteroscedastic_16_16_N{N_train}"

    F_tests = [16, 24]

    run_types = ["aleatoric_AR1", "epistemic_AR1", "both_AR1", "epistemic_fix", "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Aleatoric", "Epistemic", "Both", "Epistemic (PPE)", "Both (PPE)"]
    linestyles = ["solid"]*3 + ["dashed"]*3

    for F_test in F_tests:
        save_prefix = f"climate_F{F_test}_"
        fnames = [f"climate_F{F_test}_run{i:02d}_X_dtf" for i in range(1)]
        fnames_true = [f"climate_F{F_test}_run{i:02d}_X_dtf" for i in range(1)]
        fnames_base = [f"climate_F20_run{i:02d}_X_dtf" for i in range(1)]

        # All on one plot:
        plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
            save_prefix=save_prefix, fnames=fnames, fnames_base=fnames_base, linestyles=linestyles)

        plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
            save_prefix=save_prefix, fnames=fnames, fnames_base=fnames_base, linestyles=linestyles,
            running_mean=True, running_mean_window=10) 

        plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
            save_prefix=save_prefix, fnames=fnames, fnames_base=fnames_base, linestyles=linestyles,
            running_mean=True, running_mean_window=40) 

        plot_anomaly_distributions(params, model_name, run_types, label_names, det=True,
            save_prefix=save_prefix, fnames=fnames, fnames_base=fnames_base, linestyles=linestyles,
            running_mean=True, running_mean_window=100) 
    