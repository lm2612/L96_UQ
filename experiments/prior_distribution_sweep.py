import os
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal
from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianLinearRegression, BayesianNN, BayesianNN_Heteroscedastic
from scripts.train_bayesian import bayesian_train
from scripts.train_bayesian_mcmc import  bayesian_train_mcmc
from plotting_scripts.plot_inputs_outputs import plot_inputs_outputs
from plotting_scripts.plot_cov import plot_cov

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
training_params = {
    'N_train': 100, 
    'batch_size': 128,
    'N_timesteps': 1,
    'lr': 0.0003,
    'num_iterations' : 100000 ,
    'num_samples' : 10000 ,
    'warmup_steps' : 500000 ,
    'num_chains':1,
    'save_prefix':'',
}
N_train = training_params['N_train']
K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    
# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

# Run large sweep across multiple prior distributions and methods used to training
# Distribution for prior dist
dist_names = ["Normal", "Laplace"]
# Scale (std for Normal, scale for laplace dist)
prior_scales = [0.5, 1.0, 2.0]

# centre prior dist on zero or on params from deterministic NN training (theta_mse)
pretrained_model_name = f"NN_16_16_N{N_train}"
pretrained_dict = torch.load( f"{data_path}/{pretrained_model_name}/model.pt")
pretrained_state = pretrained_dict['model'].state_dict()

# Use Variational inference or MCMC?
training_methods = ["VI", "mcmc_RW"]
for dist_name in dist_names:
    for prior_scale in prior_scales:
        for param_dict in [None, pretrained_state]:
            if param_dict is None:
                prior_dist_str = f"{dist_name}(0,{prior_scale})"
            else:
                prior_dist_str = f"{dist_name}(theta_mse,{prior_scale})"

            model_name =  f"BayesianNN_Heteroscedastic_16_16_N{N_train}_prior{prior_dist_str}"      
            model = BayesianNN_Heteroscedastic(1, 1, [16, 16], dist_name=dist_name, 
                weight_scale=min(1., prior_scale), bias_scale=prior_scale, param_dict = param_dict) 
            for training_method in training_methods:
                if training_method == "VI":
                    # variational inference:
                    guide = AutoMultivariateNormal(model)
                    training_params["training_method"] = "VI"
                    training_params["kernel_name"] = None
                    print(f"Running {model_name} variational inference...")
                    try:
                        bayesian_train(params, training_params, model_name, model, guide)
                        plot_cov(params, training_params, model_name)
                    except ValueError as e:
                        print(f"WARNING. Failed with error {e}. Continuing...")
                else:
                    # mcmc
                    kernel_name = training_method.split("_")[-1]
                    training_params["training_method"] = "mcmc"
                    training_params["kernel_name"] = kernel_name
                    print(f"Running {model_name} mcmc with {kernel_name}...")
                    bayesian_train_mcmc(params, training_params, model_name, model, kernel_name=kernel_name)
                    plot_cov(params, training_params, model_name)



