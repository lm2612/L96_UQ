import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.animation import FuncAnimation
from IPython import display
import functools

## Plot Lorenz data on polar axis
def plot_polar(X, ax=None):
    """ Plots the Lorenz-96 variables on a polar axis """
    if isinstance(X, tuple):
        if len(X) == 2:
            t, X = X
            Y = None
        elif len(X) == 3:
            t, X, Y = X
    else:
        t, Y = None, None
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # Clear axis
    ax.clear()
    # Plot X
    theta = np.linspace(0, 2*np.pi, X.shape[0]+1)
    r_cyclic = np.concatenate((X, X[0:1]))
    ax.plot(theta, r_cyclic, color="black", lw=2 )
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(theta, 5*np.ones(theta.shape[0]), color="grey", lw=0.5)
    if Y is not None:
        # Plot Y
        r_cyclic = 15. + 2.5*np.concatenate((Y, Y[0:1]))
        theta = np.linspace(0, 2*np.pi, Y.shape[0]+1)
        ax.plot(theta, r_cyclic, color="blue", lw=2)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(theta, 15*np.ones(theta.shape[0]), color="grey", lw=0.5)

    # Fix axis manually
    ax.set_rmax(20 )
    ax.set_rmin(-10)
    ax.set_rticks([]) #range(-10, 20, 5), labels=[])  # Less radial ticks
    ax.set_xticks([])
    ax.grid(False)
    ax.set_axis_off()
    return ax

## Plot Lorenz data on cartesian axis
def plot_cartesian(X, ax=None):
    """ Plots the Lorenz-96 variables on a cartesian axis """
    if isinstance(X, tuple):
        if len(X) == 2:
            t, X = X
            Y = None
        elif len(X) == 3:
            t, X, Y = X
    else:
        t, Y = None, None
    if ax is None:
        fig, ax = plt.subplots()
    # Clear axis
    ax.clear()
    # Plot X
    theta = np.arange(0, X.shape[0])
    ax.plot(theta, X, color="black", lw=2 )
    if Y is not None:
        # Plot Y
        theta = np.linspace(0, X.shape[0], Y.shape[0])
        ax.plot(theta, Y, color="blue", lw=2)
    plt.axis(ymin=-10, ymax=15)
    if t is not None:
        plt.title(f"T={t}")
    return ax


def plot_data_gif(params, model_name="", fname_X="X_dtf", fname_Y=None, T=1000, save_prefix="", 
    plot_fn = plot_polar):
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
    if plot_fn == plot_polar:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    else:
        fig, ax = plt.subplots(figsize=(8, 3))
    # Create frames
    frames = [(t, X_truth[t]) for t in range(0, T, 1)]

    # Create animation
    anim = FuncAnimation(fig, functools.partial(plot_fn, ax=ax), 
                        frames=frames, interval=100, blit=False)
    anim.save(f"{plot_path}/{save_prefix}lorenz96_Xonly.gif") #, writer='imagemagick'
    print(f"Saved animation to {plot_path}/{save_prefix}lorenz96_Xonly.gif")


    if fname_Y is not None:
        Y_truth = np.load(f"{data_path}/{fname_Y}.npy")

        plt.clf()
        if plot_fn == plot_polar:
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        else:
            fig, ax = plt.subplots()     
        # Create frames
        frames = [(t, X_truth[t], Y_truth[t]) for t in range(0, T, 1)]
        # Create animation
        anim = FuncAnimation(fig, functools.partial(plot_fn, ax=ax), 
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

    plot_data_gif(params)    
    plot_data_gif(params, fname_Y="Y_dtf")    
