import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot


from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import add_axis_weather

def plot_spread_v_skill(params, model_name, run_types, label_names, 
    save_prefix="", ylim=3, samples_per_bin=25, num_plots=40):
    """Plots spread skill metrics """
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'
    filenames = [f'{model_path}/{run_type}_X_dtf.npy' for run_type in run_types]
    print(filenames)

    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    X_truth = np.load(f"{data_path}/X_dtf.npy")

    # Load ml param model results - must all be same size
    X_mls = np.stack([np.load(filename) for filename in filenames])
    n_ens = X_mls.shape[1]
    # Reshape for initial conditions
    T = 10
    nt = int(T/dt_f)
    N_init = X_mls.shape[2] // nt
    time = np.arange(0, T, dt_f)
    X_mls = X_mls.reshape((len(filenames), n_ens, N_init, nt, K))
    
    # truth
    N_init_truth = X_truth.shape[0] // nt 
    X_truth = X_truth.reshape(((N_init_truth, nt, K)))[:N_init]

    # Compute mean and std across ensemble (axis=1)    
    X_mean = X_mls.mean(axis=1)
    X_std = X_mls.std(axis=1)

    X_diff = (X_mean - X_truth)

    print(X_diff.shape, X_mean.shape)

    timesteps = np.linspace(dt_f, 2.0/dt_f, num_plots)

    for t in timesteps:
        t = int(t)

        n_samples = N_init * K
        print(n_samples)
        n_bins = n_samples // samples_per_bin
        print(n_bins)

        plt.clf()
        fig, axs = plt.subplots(1, 1, figsize=(6, 4), sharex=True)
        
        for r in range(len(run_types)):
            
            # Flatten to combine N_init initial conditions and K dimensions 
            X_diff_rt = X_diff[r, :, t].flatten()
            X_std_rt = X_std[r, :, t].flatten()

            # Sort into order of increasing spread
            sorted_inds = np.argsort(X_std_rt.flatten())
            X_diff_sorted = X_diff_rt[sorted_inds]
            X_std_sorted = X_std_rt[sorted_inds]
            
            # Reshape into bins
            X_diff_bins = X_diff_sorted.reshape((n_bins, samples_per_bin))
            X_std_bins = X_std_sorted.reshape((n_bins, samples_per_bin))

            # Average across variances and take std of err
            ## 1) the standard deviation of the ensemble for each bin:
            spread = X_std_bins.mean(axis=-1)
            
            ## 2) the standard deviation of the ensemble mean error for each bin 
            # Note, Leutbecher says we can approximate std with RMS error of the ensemble mean as proxy
            #igma_err = X_diff_bins.std(axis=-1)
            sigma_err = np.sqrt((X_diff_bins**2).mean(axis=-1))

            plt.scatter(spread, sigma_err, 
                color=plotcolor(run_types[r]),
                alpha=0.8,
                label=label_names[r])
        plt.legend(loc="lower right")
        plt.xlabel("r.m.s spread")
        plt.ylabel("r.m.s error")
        plt.title(f"Time = {t*dt_f:.1f}")
        plt.plot([0, ylim], [0, ylim], 'k--')
        plt.axis(xmin=0, xmax=ylim, ymin=0, ymax=ylim)


        plt.savefig(f"{plot_path}/{save_prefix}spread_v_sigma_err_{t:03d}.png")
        print(f"Saved to {plot_path}/{save_prefix}spread_v_sigma_err_{t:03d}.png")    



