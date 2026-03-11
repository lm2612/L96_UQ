import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro

from ml_models.BayesianModels import BayesianLinearRegression, BayesianNN, BayesianNN_Heteroscedastic

from plotting_scripts.compare_uncertainties import compare_uncertainties

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

model_paths = ['BayesianNN_Heteroscedastic_16_16_N100_priorNormal(0,0.5)/mcmc_RW_predictive.pt', 
    'BayesianNN_Heteroscedastic_16_16_N100_priorNormal(0,1.0)/mcmc_RW_predictive.pt', 
    'BayesianNN_Heteroscedastic_16_16_N100_priorLaplace(0,1.0)/mcmc_RW_predictive.pt', 
    'BayesianNN_Heteroscedastic_16_16_N100_priorNormal(theta_mse,1.0)/mcmc_RW_predictive.pt',
    'BayesianNN_Heteroscedastic_16_16_N100_priorNormal(0,0.5)/model_best.pt', 
    'BayesianNN_Heteroscedastic_16_16_N100_priorNormal(0,1.0)/model_best.pt', 
    'BayesianNN_Heteroscedastic_16_16_N100_priorLaplace(0,1.0)/model_best.pt', 
    'BayesianNN_Heteroscedastic_16_16_N100_priorNormal(theta_mse,1.0)/model_best.pt']

model_labels = ["MCMC \n N(0,0.5)", 
    "MCMC \n N(0,1.0)", 
    "MCMC \n L(0,1.0)",
    "MCMC \n "+r"N($\hat{\theta}$,1.0)",
    "VI \n N(0,0.5)",
    "VI \n N(0,1.0)", 
    "VI \n L(0,1.0)",
    "VI \n "+r"N($\hat{\theta}$,1.0)",
]
compare_uncertainties(params, model_paths, model_labels, save_prefix="VI_prior_", n_bins=5)
