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
    model_name =  f"BayesianNN_Heteroscedastic_16_16_N100" 
    model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
    test_params['save_model_path'] = model_path

    # Set up model
    output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
    pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
    pyro_model = output_dicts["model"]
    guide = output_dicts["guide"]

    # Lag-1 Autocorrelation of long timeseries is 0.984865 
    phi = 0.984865 
    parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, phi=phi)

    for i in range(0, 10):
        test_params['fname'] = f'run{i:02d}_X_dtf.npy'

        # Run deterministic
        # Deterministic - no uncertainty
        fixed_param_NN = pyro_model.get_fixed_param_NN(guide.median())
        def param_func(x):
            with torch.no_grad():
                mean_sigma = fixed_param_NN(x.unsqueeze(-1))
            mean, sigma = mean_sigma.chunk(2, dim=-1)
            return mean.squeeze()
        test_params['runtype'] = 'deterministic'
        test_params['save_prefix'] = f'deterministic_climate_F{F_test}_run{i:02d}_' 
        test_params['n_ens'] = 1
        test(params, test_params, param_func)
        
        test_params['n_ens'] = 50

        # Run Aleatoric with AR1 
        parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, phi=phi)
        param_func = parameterisation_AR1.aleatoric_only
        test_params['runtype'] = 'aleatoric'
        test_params['save_prefix'] = f'aleatoric_AR1_climate_F{F_test}_run{i:02d}_' 
        test(params, test_params, param_func)


        # Run Epistemic with fixed parameters - will sample guide parameters before each ensemble member
        parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, include_sigma = False, phi=0.)

        def param_sample(n):
            """ Set up parameterisation for ensemble member n """
            # Open file
            fixed_nn_model = torch.load(f"{model_path}/fixed_param_model_{n}.pt", weights_only=False)
            # Update in parameterisation_AR1
            parameterisation_AR1.fixed_param_NN = fixed_nn_model
        param_func = parameterisation_AR1.keep_epistemic_fixed

        test_params['runtype'] = 'epistemic'
        test_params['save_prefix'] = f'epistemic_fix_climate_F{F_test}_run{i:02d}_' 
        test(params, test_params, param_func, param_sample=param_sample)

        # Run Both with fixed parameters - epistemic fixed and aleatoric sampled using AR1
        parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, include_sigma=True, phi=phi)
        def param_sample(n):
            """ Set up parameterisation for ensemble member n """
            # Open file
            fixed_nn_model = torch.load(f"{model_path}/fixed_param_model_{n}.pt", weights_only=False)
            # Update in parameterisation_AR1
            parameterisation_AR1.fixed_param_NN = fixed_nn_model
        param_func = parameterisation_AR1.keep_epistemic_fixed
        
        test_params['runtype'] = 'both'
        test_params['save_prefix'] = f'both_fix_AR1_climate_F{F_test}_run{i:02d}_' 
        test(params, test_params, param_func, param_sample=param_sample)

        parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, phi=phi, aleatoric=False, epistemic=True, N=2)
        param_func = parameterisation_AR1.AR1_param
        test_params['runtype'] = 'epistemic'
        test_params['save_prefix'] = f'epistemic_AR1_climate_F{F_test}_run{i:02d}_' 
        test(params, test_params, param_func)

        parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, phi=phi, aleatoric=True, epistemic=True, N=2)
        param_func = parameterisation_AR1.AR1_param
        test_params['runtype'] = 'both'
        test_params['save_prefix'] = f'both_AR1_climate_F{F_test}_run{i:02d}_' 
        test(params, test_params, param_func)
