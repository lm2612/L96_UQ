import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pyro 
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianLinearRegression, BayesianNN, BayesianNN_Heteroscedastic

from plotting_scripts.plot_dicts import plotcolor
from utils.summary_stats import summary_stats

def compare_uncertainties(params, model_paths, model_labels, save_prefix="", n_bins=1):
    """
    Create bar plot to compare aleatoric/epistemic/total uncertainty across entire dataset.
    * params of system
    * model_path must be full model path including dir and .pt
    * model_labels to label the x-axis"""
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'

    plot_path = f'{data_path}/comparison_plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Get data 
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

    for n, model_path in enumerate(model_paths):
        print(model_path)
        output_dicts = torch.load(f"{data_path}/{model_path}")
        # If VI, load model and guide to set up predictive:
        if "guide" in output_dicts.keys():
            model = output_dicts["model"]
            guide = output_dicts["guide"]
            model_dir = model_path.split("/")[0]
            pyro.get_param_store().load( f"{data_path}/{model_dir}/pyro_best_params.pt")
            # Predictive 
            num_samples = 800
            predictive = Predictive(model, guide=guide, num_samples=num_samples,
                                    return_sites=("obs", "_RETURN"))
            
        else:
            model = output_dicts["model"]
            predictive = output_dicts["predictive"]

        # Evaluate on validation dataset
        sorted_inds = torch.sort(X_torch.flatten())[1]

        # Print bins - sanity check
        n_samples = len(sorted_inds)
        n_per_bin = n_samples // n_bins
        X_bins = X_torch.flatten()[sorted_inds].reshape((n_bins, n_per_bin))
        print(f"Plotting with X bins: ", X_bins)
        
        samples = predictive(X_torch)

        # Calculate epistemic and aleatoric and total variances
        samples_mean = samples["_RETURN"][..., 0]
        samples_sigma2 =  (torch.exp(samples["_RETURN"][..., 1])+model.eps)**2
        observed_samples = samples["obs"]
        
    
        # Mean prediction is mean of mean
        mean_pred = torch.mean(samples_mean, dim=0)
        # Aleatoric is mean of variance (integrated over all parameters)
        aleatoric_var = torch.mean(samples_sigma2 , dim=0)
        # Epistemic is variance of conditional mean (ignoring aleatoric uncertainty)
        epistemic_var = torch.var(samples_mean, dim=0)
        # Law of total variances
        total_var = aleatoric_var + epistemic_var
        
        # Breakdown by bins
        # Put into N bins to show edges of dataset (default is n=1 so the mean is taken across full dataset)
        n_samples = total_var.shape[0]
        n_per_bin = n_samples // n_bins
        total_var = total_var[sorted_inds].reshape((n_bins, n_per_bin)).mean(dim=1)
        
        aleatoric_var = aleatoric_var[sorted_inds].reshape((n_bins, n_per_bin)).mean(dim=1)
        epistemic_var = epistemic_var[sorted_inds].reshape((n_bins, n_per_bin)).mean(dim=1)

    
        delta = np.linspace(-0.25, 0.25, n_bins)
        
        for i in range(n_bins):
            ax.bar(n+delta[i], np.sqrt(total_var[i]),
                    width=.5/n_bins, alpha=0.5, facecolor=plotcolor("aleatoric"), 
                    edgecolor=plotcolor("aleatoric"), lw=2,
                    label="Aleatoric" if (n==0 and i==0) else None)
            ax.bar(n+delta[i], np.sqrt(epistemic_var[i]),
                    width=.5/n_bins, alpha=0.5, facecolor=plotcolor("epistemic"), 
                    edgecolor=plotcolor("epistemic"), lw=2,
                    label="Epistemic" if (n==0 and i==0) else None)
    
    ax.axis(xmin=-1, xmax=n+1)
    ax.legend(loc="upper left")
    ax.set_ylabel(f"Variance across dataset")
    plt.xticks(ticks=range(len(model_paths)), labels=model_labels, rotation=45 )
    plt.tight_layout()
    plt.savefig(f"{plot_path}{save_prefix}variance_comparison_{n_bins}.png")
    print(f"Saved as {plot_path}{save_prefix}variance_comparison_{n_bins}.png")
    plt.close()
