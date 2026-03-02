import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression

from L96.L96_model import L96OneLayerParam

def test(params, test_params, param_func, param_sample=None, reset_param=None):
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
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'

    F = test_params['F']
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print(f"Model path: {save_model_path}")
    print(f"Simulations will be saved under {save_model_path}/{save_prefix}")

    # Load truth data
    X_truth = np.load(f"{data_path}/{fname}")
    nt = int(T/dt_f)
    X_init_conds = X_truth[::nt]
    # Check N_init cannot be longer than length of initial conditions
    if N_init > X_init_conds.shape[0] :
        raise ValueError(f"N_init larger than the available number of initial conditions in {data_path}/X_dtf.npy. Reduce T={T} or N_init={N_init}")

    nt_total = N_init * nt
    print(f"Running model for {N_init} initial conditions, for T={T}MTU / {nt} timesteps). Total timesteps={nt_total}. ")

    # How often to save (for very long runs, may want to save every 10 timesteps or so, default is every timestep nt_save=1)
    nt_save = nt_total//save_step

    # Run each model for 10MTU
    X_all = np.zeros((n_ens, nt_save, K))
    U_all = np.zeros((n_ens, nt_save, K))
    t=0
    
    # Repeat for n_ens ensemble members
    for n in range(n_ens):
        print(f"Ensemble member {n}")
        # If running fixed parameters for epistemic uncertainty, sample parameters 
        if param_sample is not None:
            param_sample(n)
        for i in range(N_init):
            # Initialize model
            if reset_param is not None:
                reset_param()
            l96_model = L96OneLayerParam(X_0=X_init_conds[i], 
                                        param_func=param_func, 
                                        dt=dt_f, 
                                        F=F)

            # Run model
            X, U, time = l96_model.iterate(T)
            X_all[n, i*nt:(i+1)*nt, :] = X[::save_step]
            U_all[n, i*nt:(i+1)*nt, :] = U[::save_step]

    # Save results
    np.save(f"{save_model_path}/{save_prefix}X_dtf.npy", X_all)
    #np.save(f"{save_model_path}/{save_prefix}U_dtf.npy", U_all)
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
