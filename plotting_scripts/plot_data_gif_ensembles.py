import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.animation import FuncAnimation
from IPython import display
import functools

from plot_data_gif import plot_polar
from plot_dicts import plotcolor

def plot_polar_ensembles(X, ax=None, color="black"):
    """ Plots the Lorenz-96 variables on a polar axis """
    # X is ensemble of size n_ens x T x K
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # Clear axis
    ax.clear()
    # Plot X
    theta = np.linspace(0, 2*np.pi, X.shape[1]+1)
    for m in range(X.shape[0]):
        # Plot ensemble member m
        X_m = X[m]
        r_cyclic = np.concatenate((X_m, X_m[0:1]))
        ax.plot(theta, r_cyclic, color=color, lw=1, alpha=0.5 )
    X = X.mean(axis=0)
    ax.plot(theta, r_cyclic, color=color, lw=2 )
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta, 5*np.ones(theta.shape[0]), color="grey", lw=0.5)
    
    # Fix axis manually
    ax.set_rmax(20 )
    ax.set_rmin(-10)
    ax.set_rticks([]) #range(-10, 20, 5), labels=[])  # Less radial ticks
    ax.set_xticks([])
    ax.grid(False)
    ax.set_axis_off()
    return ax

def plot_data_gif_ensembles(params, model_name="", fname_X="X_dtf", fname_Y=None, T=1000, save_prefix=""):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    seed = 123
    np.random.seed(seed)

    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'
    plot_path = f'{model_path}/plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Load truth data
    X_truth = np.load(f"{model_path}{fname_X}.npy")
    n_time = X_truth.shape[0]

    plt.clf()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # Create frames
    frames = [(X_truth[:, t]) for t in range(0, T, 1)]
    # Create animation
    anim = FuncAnimation(fig, functools.partial(plot_polar_ensembles, ax=ax, color=plotcolor(fname_X)), 
                        frames=frames, interval=100, blit=False)
    anim.save(f"{plot_path}/{save_prefix}lorenz96_Xonly.gif") #, writer='imagemagick'
    print(f"Saved animation to {plot_path}/{save_prefix}lorenz96_Xonly.gif")


    if fname_Y is not None:
        Y_truth = np.load(f"{data_path}/{fname_Y}.npy")

        plt.clf()
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        # Create frames
        frames = [(X_truth[:, t], Y_truth[:, t]) for t in range(0, T, 1)]
        # Create animation
        anim = FuncAnimation(fig, functools.partial(plot_polar_ensembles, ax=ax), 
                            frames=frames, interval=100, blit=False)
        anim.save(f"{plot_path}/{save_prefix}lorenz96_X_Y.gif") #, writer='imagemagick'
        print(f"Saved animation to {plot_path}/{save_prefix}lorenz96_X_Y.gif")


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
    """
    plot_data_gif_ensembles(params, model_name="BayesianNN_Heteroscedastic_16_16_N100",
        fname_X="both_AR1_X_dtf", save_prefix="both_AR1_")
    plot_data_gif_ensembles(params, model_name="BayesianNN_Heteroscedastic_16_16_N100", 
        fname_X="aleatoric_AR1_X_dtf", save_prefix="aleatoric_AR1_")
    plot_data_gif_ensembles(params, model_name="BayesianNN_Heteroscedastic_16_16_N100", 
        fname_X="epistemic_AR1_X_dtf", save_prefix="epistemic_AR1_")
    
    """

    # Climate scenarios
    Fs = [20, 16, 24]

    for F in Fs:
        plot_data_gif_ensembles(params, model_name="BayesianNN_Heteroscedastic_16_16_N100",
            fname_X=f"both_AR1_climate_F{F}_run00_X_dtf", save_prefix=f"both_AR1_climate_F{F}_")
        plot_data_gif_ensembles(params, model_name="BayesianNN_Heteroscedastic_16_16_N100", 
            fname_X=f"aleatoric_AR1_climate_F{F}_run00_X_dtf", save_prefix=f"aleatoric_AR1_climate_F{F}_")
        plot_data_gif_ensembles(params, model_name="BayesianNN_Heteroscedastic_16_16_N100", 
            fname_X=f"epistemic_AR1_climate_F{F}_run00_X_dtf", save_prefix=f"epistemic_AR1_climate_F{F}_")
        
        ##plot_data_gif_ensembles(params, model_name="BayesianNN_Heteroscedastic_16_16_N100", 
        ##    fname_X=f"epistemic_PPE_climate_F{F}_run00_X_dtf", save_prefix=f"epistemic_PPE_climate_F{F}_")