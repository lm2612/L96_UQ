import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression

from L96.numerical_methods import RK2_step
from L96.L96_model import dX_dt_onelayer

def test(params, test_params, param_func, param_sample=None):
    """Function that does online test and saves output
    Args:
    - params
    """
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    fname, runtype, N_init, T = test_params['fname'], test_params['runtype'], test_params['N_init'], test_params['T']
    n_ens, save_step = test_params['n_ens'],  test_params['save_step']
    save_model_path, save_prefix = test_params['save_model_path'], test_params['save_prefix']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/truth/'

    F = test_params['F']
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print(f"Model path: {save_model_path}")
    print(f"Simulations will be saved under {save_model_path}/{save_prefix}")

    # Load truth data
    X_truth = torch.tensor(np.load(f"./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/{fname}"))
    nt = X_truth.shape[0]
    print(f"running for {nt} timesteps")
    

    # Run each model for 10MTU
    X_all = np.zeros((n_ens, nt, K))
    U_all = np.zeros((n_ens, nt, K))
    t=0
    
    # Repeat for n_ens ensemble members
    for n in range(n_ens):
        print(f"Ensemble member {n}")
        # If running fixed parameters for epistemic uncertainty, sample parameters 
        if param_sample is not None:
            param_sample()
        # Initialize model
        X_all[n, t, :] = X_truth[0]
        for t in range(1, nt):
            X_in = X_truth[t-1]
            U_pred = param_func(X_in)
            U_all[n, t, :] = U_pred
            X_all[n, t, :] = RK2_step(X_in, dX_dt_onelayer, dt_f, F=F, U=U_pred)
        

    # Save results
    np.save(f"{save_model_path}/{save_prefix}X_dtf.npy", X_all)
    np.save(f"{save_model_path}/{save_prefix}U_dtf.npy", U_all)
    # Save meta data about run 
    np.save(f"{save_model_path}/{save_prefix}test_params.npy", test_params, allow_pickle=True)

    print(f"Done. Saved to {save_model_path}/{save_prefix}X_dtf.npy")


if __name__ == "__main__":
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
                    'N_init': 50,
                    'save_step': 1,
                    'T':10 ,
                    'F':20                  }

    model_name =  f"BayesianNN_16_N50" 
    model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
    test_params['save_model_path'] = model_path

    # Set up model
    output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
    pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
    pyro_model = output_dicts["model"]
    guide = output_dicts["guide"]
    predictive = Predictive(pyro_model, guide=guide, num_samples=1, return_sites=("_RETURN", "obs"))

    # Run Epistemic with white noise
    def param_func(x):
        out = predictive(x.unsqueeze(-1))["_RETURN"]
        return out.squeeze()
    test_params['runtype'] = 'epistemic'
    test_params['save_prefix'] = 'epistemic_' 
    test(params, test_params, param_func)

    # Run Aleatoric with white noise
    sigma = pyro.get_param_store()['sigma']
        
    fixed_param_NN = pyro_model.get_fixed_param_NN(guide.median())
    fixed_param_NN.eval()
    def param_func(x):
        with torch.no_grad():
            mean = fixed_param_NN(x.unsqueeze(-1))
            out = pyro_model.sample_obs(mean)
        return out.squeeze()

    test_params['runtype'] = 'aleatoric'
    test_params['save_prefix'] = 'aleatoric_' 
    test(params, test_params, param_func)

    # Run both types of uncertainty 
    def param_func(x):
        out = predictive(x.unsqueeze(-1))["obs"]
        return out.squeeze()

    test_params['runtype'] = 'both'
    test_params['save_prefix'] = 'both_' 
    test(params, test_params, param_func)
