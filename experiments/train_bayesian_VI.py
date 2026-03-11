import os
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal

from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression
from scripts.train_bayesian import bayesian_train
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
    'lr': 0.002,
    'num_iterations' : 10000 ,
    'save_prefix':'',
    'training_method':'VI',
}
N_train = training_params['N_train']

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

# Choose prior distribution
dist_name = "Normal"
scale = 1.0
model_name =  f"BayesianNN_16_16_N{N_train}_prior{dist_name}(0,{scale})"  

# Set up model and guide
model = BayesianNN(1, 1, [16, 16], 
    dist_name=dist_name, weight_scale=scale, bias_scale=scale) 
guide = AutoMultivariateNormal(model)

bayesian_train(params, training_params, model_name, model, guide)
plot_inputs_outputs(params, training_params, model_name)
plot_cov(params, training_params, model_name)