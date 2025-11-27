import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import mtu_to_years
from utils.kde_plot import kde_plot

def plot_num_transitions(params, model_name, run_types, label_names, 
    save_prefix="", fnames=["X_dtf"],  det=False, save_step=1, kde=False, xmin=0.2, xmax=0.8, 
    T=-1, spinup=0, title=""):
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
    X_truth = X_truth[:, ::save_step, :]
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
    if T==-1:
        T = pred_regimes.shape[-1]
    pred_regimes = pred_regimes[..., spinup:T]
    

    # Fracion of time spent in regime 1 for each ensemble member / init cond
    n_init = pred_regimes.shape[1]
    n_ens = pred_regimes.shape[2]
    print(n_init)
    #fig, ax = plt.subplots(1, figsize=(10, 6))
    fig, ax = plt.subplots(1, figsize=(6, 2.5))




    num_transitions_truth = np.sum(np.abs(np.diff(true_regimes[..., spinup:T], axis=-1)), axis=-1)
    num_transitions_pred = np.sum(np.abs(np.diff(pred_regimes[..., spinup:T], axis=-1)), axis=-1)
    print(num_transitions_truth)
    print(np.min(num_transitions_pred, axis=(1,2)))
    
   
    


    exit()



    mean_truth = truth.mean()
    std_truth = truth.std()
    print(mean_truth, std_truth)
    ax.axvline(true_regimes[..., spinup:T].mean(), color="k", linestyle="dashed", label="Truth")
    ax.fill_betweenx(y=[ymin, ymax],
        x1=[mean_truth - std_truth, mean_truth - std_truth], 
        x2=[mean_truth + std_truth, mean_truth + std_truth],
        color="k", alpha=0.1)

    # Deterministic regime
    if det:
        X_det = np.concatenate([np.load(f'{model_path}/deterministic_{fname}.npy') for fname in fnames], axis=0)
        print(X_det.shape)

        X_transformed = np.stack([pca.transform(X_det[i]) for i in range(N_init)])
        max_pc = np.argmax(X_transformed, axis=2)
        print(max_pc)
        det_regimes = max_pc//2
        det = det_regimes[..., spinup:T].mean(axis=-1)
        mean_det = det.mean()
        ax.axvline(mean_det, color="lightblue", label="Deterministic")
        
    for r in range(len(run_types)):
        frac_time = np.zeros((n_init*n_ens))
        for i in range(n_init):
            for m in range(n_ens):
                # For each ensemble member, get proportion of time spent in regime
                frac_time[int(i*n_ens + m)] = pred_regimes[r, i, m].mean(axis=-1)
        if kde:
            #X_domain = np.arange(0.3, 0.5, 0.0025)
            pdf = kde_plot(frac_time, X_domain, bw=0.4)
            ax.plot(X_domain, pdf, 
                alpha=0.9, lw=1.5,
                label=label_names[r],
                color=plotcolor(run_types[r]) )
            
        else:
            ax.hist(frac_time, 
                bins = X_domain,
                alpha=0.4, lw=1,
                label=label_names[r],
                color=plotcolor(run_types[r]),
                density=True)
        mean_r = frac_time.mean()
        std_r = frac_time.std()

        
    if n_init > 1:
        X_domain = np.arange(0.1, 0.8, 0.01)

        frac_time = true_regimes.mean(axis=-1)
        print(frac_time)
        if kde:
            print("Skip")
        #    pdf = kde_plot(frac_time, X_domain)
        #    ax.plot(X_domain, pdf, 
        #            alpha=1.0, lw=2,
        #            label="Truth",
        #            color="black") 
            
        else:
            ax.hist(frac_time, 
                bins = X_domain,
                alpha=1.0, lw=1.5,
                label="Truth",
                color="black", 
                histtype="step",
                density=True)

    plt.legend()
    plt.xlabel("Ratio of time spent in regime $k=1$")
    mtu = int(dt_f * T * save_step)
    years = mtu_to_years(mtu)
    print(years)
    plt.title(f"{title} after T={mtu} MTU (~{years:.0f} Atmos. Years)")
    plt.ylabel("Probability density function")
    plt.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    plt.tight_layout()


    plt.savefig(f"{plot_path}/{save_prefix}_frac_time_in_reg_hist_T{mtu:04d}.png")
    print(f"{plot_path}/{save_prefix}_frac_time_in_reg_hist_T{mtu:04d}.png")
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
    model_name = f"BayesianNN_Heteroscedastic_16_16_N100"

    run_types = ["epistemic_fix", "aleatoric_AR1", "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    
    F=16
    fnames = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(10)]
    save_prefix = f"climate_F{F}_run00-09_"

    plot_num_transitions(params, model_name, run_types, label_names, 
        save_prefix=save_prefix+"kde_", fnames = fnames, kde=True,  save_step = 10,
        T=20000, title = f"F={F}", det=True)
