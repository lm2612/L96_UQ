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


def offline_errs(params, model_paths, save_prefixs=[""]):
    """
    Calculate offline metrics (RMSE, R2) on entire dataset.
    * params of system
    * model_path must be full model path including dir and .pt
    * model_labels to label the x-axis"""
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'

    plot_path = f'{data_path}/comparison_plots/'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Get data 
    X = np.load(f'{data_path}/X_dtf.npy')
    U = np.load(f'{data_path}/U_dtf.npy')
    print(f'Data loaded from {data_path}')

    # Subsample to remove correlations
    subsample = 1000 # (1 Time Units)
    X = X[::subsample]
    U = U[::subsample]

    N = X.shape[0]

    # Calc variance across validation dataset (same size for all N_train)
    features = np.ravel(X[:])   
    targets = np.ravel(U[:])   
    X_torch = torch.tensor(features, dtype=torch.float32).reshape((-1, 1))
    Y_torch = torch.tensor(targets, dtype=torch.float32).reshape((-1, 1))

    N_train = 100

    features = np.ravel(X[:N_train])   
    targets = np.ravel(U[:N_train])   

    features_val = np.ravel(X[N_train:])   
    targets_val = np.ravel(U[N_train:])    

    print(features.shape, targets.shape, features_val.shape, targets_val.shape)

    X_train = torch.tensor(features, dtype=torch.float32).reshape((-1, 1))
    Y_train = torch.tensor(targets, dtype=torch.float32).reshape((-1, 1))
    

    X_val = torch.tensor(features_val, dtype=torch.float32).reshape((-1, 1))
    Y_val = torch.tensor(targets_val, dtype=torch.float32).reshape((-1, 1))

    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    rmses_all, rmses_train, rmses_test = [], [], []
    maes_all, maes_train, maes_test = [], [], []
    R2s = []

    for n, model_path in enumerate(model_paths):
        print(model_path)
        save_prefix = save_prefixs[n]
        output_dicts = torch.load(f"{data_path}/{model_path}/{save_prefix}model_best.pt")
        model = output_dicts["model"]
        
        # Test on full dataset
        Y_pred = model(X_torch).detach()
        rmse = torch.sqrt(torch.mean((Y_pred - Y_torch)**2))
        mae = torch.mean(torch.abs(Y_pred - Y_torch))
        rmses_all.append(rmse.item())
        maes_all.append(mae.item())

        # Training dataset
        Y_pred = model(X_train).detach()
        rmse = torch.sqrt(torch.mean((Y_pred - Y_train)**2))
        mae = torch.mean(torch.abs(Y_pred - Y_train))
        rmses_train.append(rmse.item())
        maes_train.append(mae.item())

        # Test dataset
        Y_pred = model(X_val).detach()
        rmse = torch.sqrt(torch.mean((Y_pred - Y_val)**2))
        mae = torch.mean(torch.abs(Y_pred - Y_val))
        rmses_test.append(rmse.item())
        maes_test.append(mae.item())

        print(model_path, rmse, mae)
    return rmses_train, rmses_test

def offline_test(params, test_params, param_func, param_sample=None):
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

    model_name =  f"BayesianNN_16_16_N100_priorNormal(0,1.0)" 
    model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
    test_params['save_model_path'] = model_path

    # Set up model
    output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
    pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
    pyro_model = output_dicts["model"]
    guide = output_dicts["guide"]

    # For parameterisation, sample predictive directly (captures both aleatoric & epistemic)
    predictive = Predictive(pyro_model, guide=guide, num_samples=1, return_sites=("_RETURN", "obs"))
    def param_func(x):
        out = predictive(x.unsqueeze(-1))["obs"]
        return out.squeeze()
    test_params['runtype'] = 'both'
    test_params['save_prefix'] = 'short_offline_test_' 
    offline_test(params, test_params, param_func)

