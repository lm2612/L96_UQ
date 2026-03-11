import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianNN_Heteroscedastic, BayesianLinearRegression

from scripts.online_test import online_test
from scripts.Parameterisation import Parameterisation_VI_Heteroscedastic
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
F_tests = [20, 16, 24]
for F_test in F_tests:
    test_params = { 'fname':'run00_X_dtf.npy',
                    'runtype': None,
                    'save_model_path':'',
                    'save_prefix':'',
                    'n_ens': 50,
                    'N_init': 1,
                    'save_step': 10,
                    'run_type': 'epistemic',
                    'save_prefix': 'epistemic_fix_run00_',
                    'T':1000 ,
                    'F':F_test                  }

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

    # Set up Parameterisation for Heteroscedastic BNN learned via Variational Inference (VI)
    parameterisation = Parameterisation_VI_Heteroscedastic(pyro_model, guide=guide, phi=phi)

    for i in range(0, 10):
        test_params['fname'] = f'run{i:02d}_X_dtf.npy'

        # Run deterministic
        # Deterministic - no uncertainty
        param_func = parameterisation.deterministic
        test_params['runtype'] = 'deterministic'
        test_params['save_prefix'] = f'deterministic_climate_F{F_test}_run{i:02d}_' 
        test_params['n_ens'] = 1
        test(params, test_params, param_func)
        
        # Stochastic
        test_params['n_ens'] = 50

        # Run Aleatoric with AR1 
        param_func = parameterisation.AR1_param_aleatoric
        test_params['runtype'] = 'aleatoric'
        test_params['save_prefix'] = f'aleatoric_AR1_climate_F{F_test}_run{i:02d}_' 
        online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

    
        # Run Epistemic with fixed parameters - will sample guide parameters before each ensemble member
        def param_sample(n):
            # Set up parameterisation fixed parameters
            np.random.seed(n)
            torch.manual_seed(n)
            parameterisation.fixed_param_NN = parameterisation.pyro_model.get_fixed_param_NN(
                parameterisation.param_sample())

        param_func = parameterisation.fixed_param_epistemic
        test_params['runtype'] = 'epistemic'
        test_params['save_prefix'] = f'fixed_epistemic_climate_F{F_test}_run{i:02d}_' 
        online_test(params, test_params, param_func, 
            param_sample=param_sample, reset_param=parameterisation.reset_param)

        # Run Both with fixed parameters - epistemic fixed and aleatoric sampled using AR1
        param_func = parameterisation.fixed_param_both    
        test_params['runtype'] = 'both'
        test_params['save_prefix'] = f'fixed_both_AR1_climate_F{F_test}_run{i:02d}_' 
        online_test(params, test_params, param_func, 
            param_sample=param_sample, reset_param=parameterisation.reset_param)

        # Regular epistemic/both (AR1)
        param_func = parameterisation.AR1_param_epistemic
        test_params['runtype'] = 'epistemic'
        test_params['save_prefix'] = f'epistemic_AR1_climate_F{F_test}_run{i:02d}_' 
        online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)

        param_func = parameterisation.AR1_param_both
        test_params['runtype'] = 'both'
        test_params['save_prefix'] = f'both_AR1_climate_F{F_test}_run{i:02d}_' 
        online_test(params, test_params, param_func, reset_param=parameterisation.reset_param)
