import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression, FixedParamNN, FixedParamLinearRegression

from L96.L96_model import L96OneLayerParam

def test(params, test_params, model_name):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    fname, runtype, N_init, T = test_params['fname'], test_params['runtype'], test_params['N_init'], test_params['T']
    n_ens, save_step = test_params['n_ens'],  test_params['save_step']
    save_prefix = test_params['save_prefix']
    

    # Set up directories
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/truth/'
    load_model_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/{model_name}/' 

    F = test_params['F']
    save_model_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/{model_name}/' 
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print(f"Model path: {save_model_path}")
    print(f"Simulations will be saved under {save_model_path}/{save_prefix}")


    # Load ml param model: This depends on the type of the model
    if "Bayesian" in model_name:
        print(f"Running Bayesian NN - a stochastic parameterisation: {model_name} using {runtype} uncertainty, {n_ens} ensemble members")
        output_dicts = torch.load(f"{load_model_path}/model_best.pt", weights_only=False)
        pyro.get_param_store().load(f"{load_model_path}/pyro_params.pt")

        pyro_model = output_dicts["model"]
        guide = output_dicts["guide"]
        # Stochastic parameterisation
        if runtype == "epistemic":
            return_site = "_RETURN" 
            predictive = Predictive(pyro_model, guide=guide, num_samples=1,
                        return_sites=(return_site,))
        elif runtype == "both":
            return_site = "obs"
            predictive = Predictive(pyro_model, guide=guide, num_samples=1,
                        return_sites=(return_site,))
        elif runtype == "aleatoric":
            return_site = "obs"
            if isinstance(pyro_model, BayesianNN):
                fixed_param_NN = FixedParamNN(pyro_model, guide)
            elif isinstance(pyro_model, BayesianLinearRegression):
                fixed_param_NN = FixedParamLinearRegression(pyro_model, guide)
            fixed_param_NN.eval()
            predictive = Predictive(fixed_param_NN, guide=guide, num_samples=1,
                        return_sites=(return_site,))
        elif runtype == "deterministic":
            if n_ens != 1:
                warnings.warn(f"runtype not valid for deterministic run ({model_name}). You set n_ens={n_ens}. This will be ignored and only one member run.")
                n_ens = 1
            return_site = "_RETURN"
            fixed_param_NN = FixedParamNN(pyro_model, guide)
            fixed_param_NN.eval()
            predictive = Predictive(fixed_param_NN, guide=guide, num_samples=1,
                        return_sites=(return_site,))
        elif runtype == "mean":
            # Todo
            return_site = "obs"
        else:
            raise ValueError(f"{runtype} unknown, must be epistemic, aleatoric or mean.")
        
        def param_func(x):
            out = predictive(x.unsqueeze(-1))[return_site]
            return out.squeeze()

        if save_prefix is None:
            save_model_path = f'{save_model_path}/{runtype}_'

    elif "Aleatoric" in model_name:
        print(f"Stochastic run: {model_name}")
        if runtype == "epistemic":
            raise ValueError(f"Must be run either in aleatoric or determinstic mode.")
        
        output_dicts = torch.load(f"{load_model_path}/model_best.pt")
        ml_model = output_dicts["model"]
        ml_model.eval()

        if runtype == "deterministic":
            # Initialize param_func
            def param_func(x):
                with torch.no_grad():
                    pred = ml_model(x.unsqueeze(-1))
                # Split into mean and variance
                mean, std = pred.chunk(2, dim=-1)
                return mean.squeeze()
            if save_prefix is None:
                save_model_path = f'{save_model_path}/{runtype}_'

        else:
            # Initialize param_func
            def param_func(x):
                with torch.no_grad():
                    pred = ml_model(x.unsqueeze(-1))
                # Split into mean and variance
                mean, std = pred.chunk(2, dim=-1)
                out = torch.normal(mean=mean.squeeze(), std=std.squeeze())
                return  out
    elif "Dropout" in model_name:
        print(f"Dropout run: {model_name}")
        output_dicts = torch.load(f"{load_model_path}/model_best.pt")
        ml_model = output_dicts["model"]
        if runtype == "epistemic":
            # Use training mode rather than eval mode to add stochasticity!
            ml_model.train()
        elif runtype == "deterministic":
            ml_model.eval()
            n_ens = 1
            if save_prefix is None:
                save_model_path = f'{save_model_path}/{runtype}_'
        else:
            raise ValueError(f"Must be run either in epistemic or deterministic mode.")


        # Initialize param_func
        def param_func(x):
            with torch.no_grad():
                out = ml_model(x.unsqueeze(-1))
            return  out.squeeze()
    else:
        print(f"Deterministic run: {model_name}")
        if runtype != "":
            warnings.warn(f"runtype not valid for deterministic run. You set runtype={runtype}. This will be ignored.")
            runtype = ""
        if n_ens != 1:
            warnings.warn(f"runtype not valid for deterministic run ({model_name}). You set n_ens={n_ens}. This will be ignored and only one member run.")
            n_ens = 1

        if model_name == "OneLayer":
            print("Running single layer model with no parameterisation")
            # run with zero parameterisation
            # Initialize param_func
            def param_func(x):
                return  np.zeros_like(x)
        else:
            output_dicts = torch.load(f"{load_model_path}/model_best.pt")
            ml_model = output_dicts["model"]
            ml_model.eval()

            # Initialize param_func
            def param_func(x):
                with torch.no_grad():
                    out = ml_model(x.unsqueeze(-1))
                return  out.squeeze()

    # Load truth data
    X_truth = np.load(f"./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/{fname}")
    nt = int(T/dt_f)
    X_init_conds = X_truth[::nt]
    # Check N_init cannot be longer than length of initial conditions
    if N_init > X_init_conds.shape[0] :
        raise ValueError(f"N_init larger than the available number of intiial conditions in {data_path}/X_dtf.npy. Reduce T={T} or N_init={N_init}")

    nt_total = N_init * nt
    print(f"Running model for {N_init} initial conditions, for T={T}MTU / {nt} timesteps). Total timesteps={nt_total}. ")

    # How often to save (for very long runs, may want to save every 10 timesteps or so, default is every timestep nt_save=1)
    nt_save = nt_total//save_step

    #TODO: Tmax e.g., restart the run every 1000 MTU to avoid memory issues]?

    # Run each model for 10MTU
    X_all = np.zeros((n_ens, nt_save, K))
    U_all = np.zeros((n_ens, nt_save, K))
    t=0
    
    for i in range(N_init):
        print(f"Initial condition {i}")
        # Repeat for n_ens ensemble members (n_ens = 1 if deterministic)
        for n in range(n_ens):
            # Initialize model
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
    np.save(f"{save_model_path}/{save_prefix}U_dtf.npy", U_all)

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
                    'save_prefix':'',
                    'n_ens': 50,
                    'N_init': 1,
                    'save_step': 1,
                    'T':10 ,
                    'F':20                  }
    N_train = 50


    #model_name = "NN_2layer_N50"
    #test(params, test_params, model_name)
    #model_name = f"AleatoricExpNN_2layer_N{N_train}"
    #test_params['runtype'] = 'aleatoric'
    #test_params['save_prefix'] = '' 
    #test(params, test_params, model_name)
    
    model_name =  f"BayesianNN_multivariatefull_32_N{N_train}" 
    test_params['runtype'] = 'deterministic'
    test_params['save_prefix'] = 'longrun_deterministic_' 
    #test(params, test_params, model_name)

    test_params['runtype'] = 'epistemic'
    test_params['save_prefix'] = 'epistemic_' 
    test(params, test_params, model_name)

    test_params['runtype'] = 'aleatoric'
    test_params['save_prefix'] = 'aleatoric_' 
    test(params, test_params, model_name)

    test_params['runtype'] = 'both'
    test_params['save_prefix'] = 'both_' 
    test(params, test_params, model_name)
