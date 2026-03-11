import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.BayesianModels import BayesianNN_Heteroscedastic

from scripts.online_test import online_test
from scripts.Parameterisation import *

from plotting_scripts.plot_rmse import plot_error_trajectories

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

test_params = { 'fname':'X_dtf.npy',
                'runtype': None,
                'save_model_path':'',
                'save_prefix':'',
                'n_ens': 50,
                'N_init': 1,
                'save_step': 1,
                'T':10 ,
                'F':20                  }

# Model name
model_name =  f"BayesianNN_Heteroscedastic_16_16_N100_priorNormal(0,1.0)" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
test_params['save_model_path'] = model_path

# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]

# Lag-1 Autocorrelation of long timeseries is 0.984865 
phi = 0.984865 

np.random.seed(123)
torch.manual_seed(123)

# Loop over N
Ns = [2, 4, 8, 16, 32]

for N in Ns:
    # Set up Parameterisation for Heteroscedastic BNN learned via Variational Inference (VI)
    parameterisation = Parameterisation_VI_Heteroscedastic(pyro_model, guide=guide, phi=phi, N=N)

    # Epistemic
    param_func = parameterisation.AR1_param_epistemic
    test_params['runtype'] = 'epistemic'
    test_params['save_prefix'] = f'AR1_epistemic_N{N}_' 
    online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

# Set up model and types of simulations to plot
run_types = [f"AR1_epistemic_N{N}" for N in Ns] 
label_names = [f"N={N}" for N in Ns] 
colors = plt.cm.gnuplot_r(np.linspace(0.1, 1., 5))  
save_prefix = "N"

plot_error_trajectories(params, model_name, run_types, label_names, 
    save_prefix=save_prefix, colors=colors, ymax=10, xmax=2)