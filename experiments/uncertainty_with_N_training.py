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
    'N_train': 50, 
    'batch_size': 128,
    'N_timesteps': 1,
    'lr': 0.002,
    'num_iterations' : 10000 ,
}

for N_train in [20, 40, 60, 80, 120, 140, 160, 180]:
    training_params['N_train'] = N_train
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)


    model_name =  f"BayesianNN_16_16_N{N_train}"      # Choose LinearRegression or NN 
    model = BayesianNN(1, 1, [16, 16]) 
    guide = AutoMultivariateNormal(model)

    bayesian_train(params, training_params, model_name, model, guide)

