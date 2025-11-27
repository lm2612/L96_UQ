# Store fixed parameters/model for epistemic PPE
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianNN_Heteroscedastic, BayesianLinearRegression

from scripts.online_test import test
from scripts.AR_parameterisation import ParameterisationAR1_Heteroscedastic
from utils.concat_files import concat_files

# Set up parameters for simulation
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

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

model_name =  f"BayesianNN_Heteroscedastic_16_16_N100" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"

# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]

# Lag-1 Autocorrelation of long timeseries is 0.984865 
phi = 0.984865 
parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, include_sigma = False, phi=0.)
n_ens = 50
for n in range(n_ens):
    parameterisation_AR1.sample_guide_params()
    print([p for p in parameterisation_AR1.fixed_param_NN.parameters()])
    # Store fixed parameter NN to file
    torch.save(parameterisation_AR1.fixed_param_NN, f"{model_path}/fixed_param_model_{n}.pt")

