import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero


import torch
import pickle
from sklearn.decomposition import PCA

from plotting_scripts.plot_dicts import plotcolor
from utils.add_time_axis import mtu_to_years, mtu_to_days, add_axis_climate, add_axis_weather
from utils.kde_plot import kde_plot

plt.rcParams.update({'font.size': 14})

def plot_weather_vs_climate_graphical(params, model_name, run_types_weather, run_types_climate, label_names, 
    save_prefix="", fnames_climate=["X_dtf"], save_step=1,
    T_climate=100, spinup=0,
    zorder_weather = [1,2,3], zorder_climate=[1,2,3]):
    """Climate predictions"""
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    
    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'

    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    quantiles = np.arange(0.1, 1.05, 0.1)
    quantiles = [0.84, 0.16]
    print(quantiles, len(quantiles))

    #fig, ax = plt.subplots(1, figsize=(10, 6))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), dpi=1500)



    ### WEATHER 
    ax_weather = axs[0]
    plt.sca(ax_weather)


    # Load data
    model_path = f'{data_path}/{model_name}/'
    filenames = [f'{model_path}/{run_type}_X_dtf.npy' for run_type in run_types_weather]
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

    print(X_mls.shape)

    X_mean = X_mls.mean(axis=1)
    T_weather = 40
    timeseries = np.arange(0, 400, 10)
    timeseries_mtu = timeseries * dt_f
    xmin = -1.
    xmax = 1.
    X_domain = np.arange(xmin, xmax, 0.01)
    for r in range(len(run_types_weather)):
        print(X_mls[r].shape, X_truth.shape)
        #X_minus_ens_mean = X_mls[r] -  X_truth
        X_minus_ens_mean = X_mls[r] - X_mean[r]
        # Plot PDF of  X_minus_ens_mean at time = T= 5 days
        X_plot = X_minus_ens_mean[..., timeseries, :]
        X_plot = np.transpose(X_plot, (2,0,1,3)).reshape((len(timeseries), n_ens*N_init*K))
        print(X_plot.shape)
        X_m = X_plot.mean(axis=-1)
        X_m = np.quantile(X_plot, 0.5, axis=-1)
        X_s = X_plot.std(axis=-1)
        X_q = np.quantile(X_plot, quantiles, axis=-1)
        X_mean_r = X_mean[r][..., timeseries, :]
        print(X_mean_r.shape)
        X_mean_reshaped = np.transpose(X_mean_r, (1, 0, 2)).reshape((len(timeseries), N_init*K))
        X_mean_m = X_mean_reshaped.mean(axis=-1)
        print(X_mean_reshaped.shape)
        print(X_plot.shape, X_mean_r.shape)
        ax_weather.plot(timeseries_mtu, X_m + X_mean_m,    
                alpha=0.9, lw=2,
                label=label_names[r],
                color=plotcolor(run_types_weather[r]), zorder = zorder_weather[r])
        """
        ax_weather.fill_between(x=timeseries_mtu, 
                y1=X_m + X_mean_m - X_s, y2 = X_m + X_mean_m + X_s,
                alpha=0.3, lw=1,
                color=plotcolor(run_types_weather[r]), zorder = zorder_weather[r])"""
        print(X_q.shape)
        for j in range(len(quantiles)//2):
            print(j, (j+1)*0.1)
            ax_weather.fill_between( timeseries_mtu, 
                y1=
                X_q[ -(j+1)] + X_mean_m,
                y2= (X_q[j] + X_mean_m) ,
                lw = 1,
                color=plotcolor(run_types_weather[r]), alpha=0.5,
                zorder = zorder_weather[r])
       
        #pdf = kde_plot(X_plot, X_domain, bw=0.4)
        """ax_weather.plot(pdf, X_domain,
                alpha=0.9, lw=2,
                label=label_names[r],
                color=plotcolor(run_types_weather[r]),
                zorder = zorder_weather[r])

        ax_weather.fill_betweenx(y=X_domain, 
            x1=np.zeros_like(X_domain), x2=pdf,
            color=plotcolor(run_types_weather[r]), alpha=0.5, 
            zorder = zorder_weather[r])"""
    print("MTU:", mtu_to_days(dt_f * timeseries))
    #mtu = dt_f * timeseries
    days = mtu_to_days(timeseries_mtu)
    print(timeseries_mtu, days)
    plt.xticks([0, 1, 2])
    plt.yticks([0, 2, 4, 6, 8])

    ax_weather.legend(loc="upper left", fontsize=16)
    #plt.xlabel("Distance of ensemble members from ensemble mean")
    #plt.title(f"Weather PDF. (T={mtu:.1f} MTU, ~{days:.0f} Atmos. Days)")
    #plt.xlabel("Probability density function")
    #plt.axis(ymin=xmin, ymax=xmax) #, ymin=ymin, ymax=ymax)
    plt.axis(ymin = 0, ymax=8)
    plt.tight_layout()
    plt.axis("off")

    plt.title(f"Weather Timescales", fontsize=16)
    
        
    ax2 = add_axis_weather(ax_weather, max_days=10, step_days=5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    #ax_weather.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.plot(1, -1.23, ">k", transform=ax2.get_yaxis_transform(), 
        clip_on=False, markersize=10)



    # Done weather


    ### CLIMATE

    # Load truth data - climate
    X_truth = np.stack([np.load(f'{data_path}/{fname}.npy') for fname in fnames_climate])
    print(X_truth.shape)
    X_truth = X_truth[:, ::save_step, :]
    print(X_truth.shape)

    
    # Load ml param model results - must all be same size
    X_mls = np.stack([[np.load(f'{model_path}/{run_type}_{fname}.npy') for fname in fnames_climate] for run_type in run_types_climate])
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

    pred_regimes = []
    n_ens = 50
    pred_regimes = np.zeros((len(run_types_climate), N_init, n_ens, len_time))
    for r in range(len(run_types_climate)):
        for i in range(N_init):
            for m in range(n_ens):
                X_transformed = pca.transform(X_mls[r, i, m])
                max_pc = np.argmax(X_transformed, axis=1)
                pred_regimes[r, i, m, :] = max_pc//2

    time_inds = range(100, len_time, 1)
    time = np.arange(100*dt_f, len_time * dt_f * save_step, 1 * dt_f * save_step)[:len(time_inds)]
    print(len(time_inds), time.shape, pred_regimes.shape, true_regimes.shape)

    #
    

    # Fracion of time spent in regime 1 for each ensemble member / init cond
    n_init = pred_regimes.shape[1]
    n_ens = pred_regimes.shape[2]
    print(n_init)

    
    # PLOT CLIMATe
    
    ax_climate = axs[1]
    plt.sca(ax_climate)


    X_domain = np.arange(0.2, 0.6, 0.004)
    ymin = 0.
    ymax =40.
    xmin=0.2
    xmax=0.55

    # Subset inds
    Ts = np.arange(300, 20000, 1000)
    T_mtu = Ts * dt_f * save_step
    for r in range(len(run_types_climate)):
        pred_mean = np.zeros((len(Ts)))
        pred_std = np.zeros((len(Ts)))
        pred_quantiles = np.zeros((len(Ts), len(quantiles)))
        for t, T in enumerate(Ts):
            pred = pred_regimes[r, ..., spinup:T].mean(axis=-1)
            pred_mean[t] = np.quantile(pred, 0.5) #pred.mean()
            pred_std[t] = pred.std()
            pred_quantiles[t] = np.quantile(pred, quantiles)
        ax_climate.plot(T_mtu, pred_mean, color=plotcolor(run_types_climate[r]), 
            alpha=0.9, lw=2, label=label_names[r], zorder = zorder_climate[r])
        print("Q", len(quantiles)//2)
        for j in range(len(quantiles)//2):
            print(j, (j+1)*0.1)
            ax_climate.fill_between(T_mtu, 
                y1=pred_quantiles[:, -(j+1)],
                y2= pred_quantiles[:, j] ,
                lw = 1,
                color=plotcolor(run_types_climate[r]), alpha=0.5,
                zorder = zorder_climate[r])
        

            
            
    #plt.legend(loc="upper right")
    #plt.xlabel("Ratio of time spent in regime $k=1$")
    mtu = int(dt_f * T_climate * save_step)
    years = mtu_to_years(mtu)
    print(mtu, years)
    
    plt.title(f"Climate Timescales" , fontsize=16)
    plt.yticks([0.3, 0.35, 0.4, 0.45])

    #plt.title(f"Climate PDF. (T={mtu} MTU, ~{years:.0f} Atmos. Years)")
    #plt.ylabel("Probability density function")
    #plt.axis(xmin=xmin, xmax=xmax)
    ax_climate.axis("off")
    plt.axis(ymin = 0.28, ymax=0.48)
    plt.tight_layout()


    ax2 = add_axis_climate(ax_climate) 
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.plot(1, 0.24999, ">k", transform=ax2.get_yaxis_transform(), 
        clip_on=False, markersize=10)



    plt.savefig(f"{plot_path}/graphical_abstract_shading.png", dpi=1500)
    print(f"{plot_path}/graphical_abstract_shading.png")
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

    run_types_weather = ["new_epistemic_AR1", "aleatoric_AR1"] #, "new_both_AR1"]
    zorder_weather = [10, 5, 1]


    run_types_climate = ["epistemic_fix", "aleatoric_AR1"] #, "both_fix_AR1"] #, "aleatoric",] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    
    F=20
    fnames_climate = [f"climate_F{F}_run{i:02d}_X_dtf" for i in range(10)]
    zorder_climate = [5, 10, 1]


    plot_weather_vs_climate_graphical(params, model_name, run_types_weather, run_types_climate, 
        label_names,  fnames_climate = fnames_climate, save_step = 10, T_climate=4000, 
        zorder_weather=zorder_weather, zorder_climate=zorder_climate)
