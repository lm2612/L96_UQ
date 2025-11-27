import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import mtu_to_years
from utils.kde_plot import kde_plot


def plot_regime_uncertainty_distributions_calibration(params, model_name, 
    run_type_calibration, fnames_true_calibration, run_types, label_names, 
    run_types_other = [], label_names_other = [],
    save_prefix="", fnames=["X_dtf"], save_step=1, xmin=0.2, xmax=0.8, 
    T_calib=2000, T_plot=20000, spinup=0, title=""):
    """Climate distributions after calibration"""
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

    X_truth_calib =  np.stack([np.load(f'{data_path}/{fname}.npy') for fname in fnames_true_calibration])
    X_truth_calib = X_truth_calib[:, ::save_step, :]
    
    # Load ml param model results - must all be same size
    X_mls = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames] for run_type in run_types])
    
    print(X_mls.shape)
    n_ens = X_mls.shape[2]
    N_init = X_mls.shape[1]
    len_time = X_mls.shape[3]

    # load other ml results to plot (no calibration!)
    X_mls_other = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames] for run_type in run_types_other])


    # Load calibration param
    X_calib = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames_true_calibration] for run_type in run_types]) 
    
    # Load PCA object
    pca = np.load(f"{data_path}/pca_fit.npy", allow_pickle=True).item()
    print(pca)

    ## How often is our simulation in each 'regime' over the entire timeseries - look for dominant PCs
    # Truth
    X_transformed = np.stack([pca.transform(X_truth[i]) for i in range(N_init)])
    max_pc = np.argmax(X_transformed, axis=2)
    true_regimes = max_pc//2
    true_regime_wn1 = np.sum(true_regimes==0)
    true_regime_tot = true_regimes.shape[0]*true_regimes.shape[1]
    
    # Truth calib
    X_transformed = np.stack([pca.transform(X_truth_calib[i, spinup:T_calib]) for i in range(N_init)])
    max_pc = np.argmax(X_transformed, axis=2)
    true_calib_regimes = max_pc//2


    # Pred    
    pred_regimes = np.zeros((len(run_types), N_init, n_ens, len_time))
    calib_regimes = np.zeros((len(run_types), N_init, n_ens, T_calib))
     
    for r in range(len(run_types)):
        for i in range(N_init):
            for m in range(n_ens):
                X_transformed = pca.transform(X_mls[r, i, m])
                max_pc = np.argmax(X_transformed, axis=1)
                pred_regimes[r, i, m, :] = max_pc//2

                X_transformed = pca.transform(X_calib[r, i, m, spinup:T_calib])
                max_pc = np.argmax(X_transformed, axis=1)
                calib_regimes[r, i, m, :] = max_pc//2
    
    other_regimes = np.zeros((len(run_types_other), N_init, n_ens, len_time))
    for r in range(len(run_types_other)):
        for i in range(N_init):
            for m in range(n_ens):
                X_transformed = pca.transform(X_mls_other[r, i, m])
                max_pc = np.argmax(X_transformed, axis=1)
                other_regimes[r, i, m, :] = max_pc//2


    time_inds = range(100, len_time, 1)
    time = np.arange(100*dt_f, len_time * dt_f * save_step, 1 * dt_f * save_step)[:len(time_inds)]
    print(len(time_inds), time.shape, pred_regimes.shape, true_regimes.shape)

    # Let's calibrate up to T_calib and first ensemble member only and epistemic only (first index)
    #calib_regimes = pred_regimes[:, 4, :, spinup:T_calib].mean(axis=-1)
    #calib_regimes = pred_regimes[:, :, :, spinup:T_calib].mean(axis=(1,3))
    print(calib_regimes.shape)
    calib_regimes = calib_regimes.mean(axis=(1,3))
    
    # Get truth
    #true_calib_regimes = true_regimes[:, spinup:T_calib]
    truth = true_calib_regimes.mean(axis=-1)
    mean_truth = truth.mean()
    std_truth = truth.std()
    
    # Get indices for each runtype
    indices_all = []

    for r in range(len(run_types)):
        mask = (calib_regimes[r] > mean_truth - std_truth) & (calib_regimes[r] < mean_truth + std_truth)
        indices = np.where(mask)[0]

        print(indices)
        print(mean_truth - std_truth,  mean_truth + std_truth)
        mtu = int(dt_f * T_calib * save_step)
        print(f"at T={mtu}MTU")
        print(f"{len(indices)} simulations found that fall between {mean_truth - std_truth:.3f} and {mean_truth + std_truth:.3f}")
        print(calib_regimes[r, indices])

        indices_all.append(indices)

    fig, ax = plt.subplots(1, figsize=(6, 2.5))
    X_domain = np.arange(0.1, 0.8, 0.004)
    ymin = 0.
    ymax= 80.

    truth = true_regimes[:, spinup:T_plot].mean(axis=-1)
    mean_truth = truth.mean()
    std_truth = truth.std()

    ax.axvline(true_regimes[..., spinup:T_plot].mean(), color="k", linestyle="dashed", label="Truth")

    ax.fill_betweenx(y=[ymin, ymax],
        x1=[mean_truth - std_truth, mean_truth - std_truth], 
        x2=[mean_truth + std_truth, mean_truth + std_truth],
        color="k", alpha=0.1)
    for r in range(len(run_types)):
        indices = indices_all[r]
        n_keep = len(indices)

        frac_time = np.zeros((N_init*n_ens))
        frac_time_calibrated = np.zeros((N_init*n_keep))
        for i in range(N_init):
            m_keep = 0
            for m in range(n_ens):
                # For each ensemble member, get proportion of time spent in regime
                frac_time[int(i*n_ens + m)] = pred_regimes[r, i, m, spinup:T_plot].mean(axis=-1)
                if m in indices:
                    frac_time_calibrated[int(i*n_keep + m_keep)] = pred_regimes[r, i, m, spinup:T_plot].mean(axis=-1)
                    m_keep += 1
        
        pdf = kde_plot(frac_time, X_domain, bw=0.4)
        pdf_calibrated = kde_plot(frac_time_calibrated, X_domain, bw=0.4)
        """ax.hist(frac_time, 
                bins = X_domain,
                alpha=0.1, lw=1,
                label=label_names[r],
                color=plotcolor(run_types[r]),
                density=True)
        ax.hist(frac_time_calibrated, 
                bins = X_domain,
                alpha=0.4, lw=1,
                label=label_names[r],
                color=plotcolor(run_types[r]),
                density=True)"""
        
        ax.plot(X_domain, pdf, 
            alpha=0.4, lw=1, # linestyle="dashed",
            label="All "+label_names[r],
            color=plotcolor(run_types[r]) )
        ax.plot(X_domain, pdf_calibrated, 
            alpha=1.0, lw=1.5, 
            label="Constrained "+label_names[r],
            color=plotcolor(run_types[r]) )
            
    for r in range(len(run_types_other)):
        frac_time = np.zeros((N_init*n_ens))
        for i in range(N_init):
            for m in range(n_ens):
                # For each ensemble member, get proportion of time spent in regime
                frac_time[int(i*n_ens + m)] = other_regimes[r, i, m, spinup:T_plot].mean(axis=-1)
        pdf = kde_plot(frac_time, X_domain, bw=0.4)
        ax.plot(X_domain, pdf, 
            alpha=0.4, lw=1, # linestyle="dashed",
            label=label_names_other[r],
            color=plotcolor(run_types_other[r]) )

    plt.legend()
    plt.xlabel("Ratio of time spent in regime $k=1$")
    mtu = int(dt_f * T_plot * save_step)
    years = mtu_to_years(mtu)
    print(years)
    plt.title(f"{title} after T={mtu} MTU (~{years:.0f} Atmos. Years)")
    plt.ylabel("Probability density function")
    plt.axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    plt.tight_layout()

    plt.savefig(f"{plot_path}/{save_prefix}_calibrated_frac_time_in_reg_hist_T{mtu:04d}.png")
    print(f"{plot_path}/{save_prefix}_calibrated_frac_time_in_reg_hist_T{mtu:04d}.png")
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

    run_types = ["epistemic_fix", "both_fix_AR1"] #,  ["epistemic_fix", "both_fix_AR1", ...]
    label_names = [ "Epistemic", "Both"]
    
 
    run_type_calibration = ["epistemic_fix_climate_F20_run04_X_dtf", "both_fix_AR1_climate_F20_run04_X_dtf"]

    fnames_true_calibration = [f"climate_F20_run{i:02d}_X_dtf" for i in range(10)]

    Fs=[20, 16, 24]
    for F in Fs:
        fnames = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(10)]
        save_prefix = f"climate_F{F}_run00-09_"
        for r in range(len(run_types)):
            plot_regime_uncertainty_distributions_calibration(params, model_name, 
                [run_types[r]], fnames_true_calibration, [run_types[r]], [label_names[r]], 
                save_prefix=save_prefix+run_types[r], fnames = fnames, save_step = 10,
                T_calib = 6000, title = f"F={F}")

