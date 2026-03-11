import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


import torch
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoMultivariateNormal, AutoLowRankMultivariateNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam

from ml_models.BayesianModels import BayesianLinearRegression, BayesianNN
from utils.summary_stats import summary_stats


def plot_cov(params, training_params, model_name, num_samples=1000):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f,  = params['dt'], params['dt_f']
    N_train = training_params['N_train']
    training_method = training_params['training_method']
    save_prefix = training_params['save_prefix']
    # Set up directory
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    model_path = f'{data_path}/{model_name}/'

    if training_method == "VI":
        # Variational inference used
        kernel_name = ""
        output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
        pyro_model = output_dicts["model"]
        guide = output_dicts["guide"]
        
        pyro.get_param_store().load( f"{model_path}/pyro_best_params.pt")

        # Sample parameters from guide
        sample_dicts = [guide() for _ in range(num_samples)]  # list of OrderedDicts

        # Stack each parameter: shape becomes (num_samples, *param_shape)
        keys = ["layers.0.bias", "layers.0.weight", "layers.1.bias" ,"layers.1.weight",
            "layers.2.bias", "layers.2.weight"] # same order as MCMC keys. otherwise use sample_dicts[0].keys()
        posterior_samples = OrderedDict( (k, torch.stack([sd[k].detach() for sd in sample_dicts], dim=0))
            for k in keys
        )
    else:
        # MCMC used - more flexible for capturing full distribution
        kernel_name = training_params["kernel_name"]
        output_dicts = torch.load(f"{model_path}/mcmc_{kernel_name}_predictive.pt", weights_only=False)
        pyro_model = output_dicts["model"]
        predictive = output_dicts["predictive"]
        num_samples = predictive.num_samples
        posterior_samples = output_dicts["samples"]
        
    # Stack all parameters into one torch tensor and compute shapes for each bias/weight in layer
    params_all = torch.hstack([torch.flatten(v, start_dim=1) for v in posterior_samples.values()])
    param_size_list = [torch.flatten(v, start_dim=1).shape[1] for k, v in posterior_samples.items()]
    param_size_dict = {k: torch.flatten(v, start_dim=1).shape[1] for k, v in posterior_samples.items()} 
    param_names = list(posterior_samples.keys())
    total_params = params_all.shape[1]

    # Compute covariance matrix
    cov = params_all.T.cov().T
    
    # Colorbar: max of cov is always >1, scale colorbar by min
    vmin = cov.min()
    
    # Plot
    plt.clf()
    fig, ax = plt.subplots(1)
    x = np.arange(0, total_params)
    plt.pcolormesh(x, x, cov, cmap="RdBu_r", vmax=np.abs(vmin), vmin=vmin)
    param_divider = 0
    text_pos = 0
    for i in range(len(param_size_list)):
        text_pos = param_divider + param_size_list[i]//2
        param_divider += param_size_list[i]

        plt.axhline(param_divider, color='k', linestyle="dashed", alpha=0.2)
        plt.axvline(param_divider, color='k', linestyle="dashed", alpha=0.2)

        if i==len(param_size_list)-2:
            va = 'bottom'
        else :
            va = 'top'
        
        plt.text(-1., text_pos, param_names[i], va='center', ha='right')
        plt.text(text_pos, -1., param_names[i], va='top', ha='right', rotation=45)
    
    # Add mean values along vertical axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.axis(xmin=-5, xmax=total_params+2, ymin = -5, ymax = total_params+2)
    plt.tight_layout()
    plt.savefig(f"{model_path}/{training_method}{kernel_name}_cov.png")
    print(f"Saved as {model_path}/{training_method}{kernel_name}_cov.png")

