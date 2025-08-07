import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN
from L96.L96_model import L96OneLayerParam
from utils.kde_plot import kde_plot

from plotting_scripts.plot_dicts import plotcolor


def plot_distributions(params, model_name, run_types, label_names, save_prefix="", linestyles = None):
    """Plots distributions """
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

    # Load ml param model results
    X_mls = [np.load(filename) for filename in filenames]

    # Plot distributions
    fig = plt.figure(figsize=(8, 5), dpi=100)
    X_domain = np.linspace(-15., 20., 100)
    for r in range(len(run_types)):
        pdf = kde_plot(X_mls[r], X_domain)
        plt.plot(X_domain, pdf, 
                label=label_names[r],
                color=plotcolor(run_types[r]),
                alpha=0.6, 
                linestyle=linestyles[r] if linestyles is not None else "solid")
    pdf_truth = kde_plot(X_truth[:], X_domain)
    plt.plot(X_domain, pdf_truth, color="black", label="Truth")

    plt.legend()
    plt.savefig(f"{plot_path}{save_prefix}X_pdf.png")
    print(f"Saved to {plot_path}{save_prefix}X_pdf.png")
    plt.gca().set_yscale('log')
    plt.savefig(f"{plot_path}{save_prefix}X_log_pdf.png")
    print(f"Saved to {plot_path}{save_prefix}X_log_pdf.png")


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
    run_types = ["epistemic", "aleatoric", "both"] # Or run_types = ["epistemic_fix", "aleatoric_AR1_", ...]
    label_names = [ "Epistemic", "Aleatoric", "Both"]
    save_prefix = "whitenoise_"

    plot_distributions(params, model_name, run_types, label_names, save_prefix=save_prefix)


