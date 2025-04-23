import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import FixedParamNN

from L96.L96_model import L96OneLayerParam

# Get paths
# Define dimensions of system (fixed)
K = 8   
J = 32  

# Define the "true" parameters
h = 1
F = 20  # 8
c = 10
b = 10

# Define time-stepping, random seed
dt = 0.001
dt_f = dt * 5
seed = 123
np.random.seed(seed)

N_train = 50
model_name = f"BayesianNN_2layer_N{N_train}"      # Choose LinearRegression or NN  or OneLayer
#model_name = f"BayesianLinearRegression_N{N_train}"
#model_name = f"BayesianNN_2layer_N{N_train}"
#model_name = f"BayesianLinearRegression_N{N_train}"
runtype = "both"    # epistemic, aleatoric or None
n_ens = 50               # number of times to run for (for deterministic this will be 1) (quite slow when running large ensembles eg 50)
# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_path = f'{data_path}/{model_name}/'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
print(save_model_path)


# Load ml param model: This depends on the type of the model
if "Bayesian" in model_name:
    print(f"Running Bayesian NN - a stochastic parameterisation: {model_name} using {runtype} uncertainty, {n_ens} ensemble members")
    output_dicts = torch.load(f"{save_model_path}/model_best.pt")
    pyro.get_param_store().load(f"{save_model_path}/pyro_params.pt")

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
        fixed_param_NN = FixedParamNN(pyro_model, guide)
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

    save_model_path = f'{save_model_path}/{runtype}_'
elif "Aleatoric" in model_name:
    print(f"Stochastic run: {model_name}")
    if runtype == "epistemic":
        raise ValueError(f"Must be run either in aleatoric or determinstic mode.")
    
    output_dicts = torch.load(f"{save_model_path}/model_best.pt")
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
        save_model_path = f'{save_model_path}/{runtype}_'

    else:
        # Initialize param_func
        def param_func(x):
            with torch.no_grad():
                pred = ml_model(x.unsqueeze(-1))
            # Split into mean and variance
            mean, std = pred.chunk(2, dim=-1)
            out = np.random.normal(loc=mean.squeeze(), scale=std.squeeze())
            return  out
elif "Dropout" in model_name:
   
    print(f"Dropout run: {model_name}")
    output_dicts = torch.load(f"{save_model_path}/model_best.pt")
    ml_model = output_dicts["model"]
    if runtype == "epistemic":
        # Use training mode rather than eval mode to add stochasticity!
        ml_model.train()
    elif runtype == "deterministic":
        ml_model.eval()
        n_ens = 1
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
        output_dicts = torch.load(f"{save_model_path}/model_best.pt")
        ml_model = output_dicts["model"]
        ml_model.eval()

        # Initialize param_func
        def param_func(x):
            with torch.no_grad():
                out = ml_model(x.unsqueeze(-1))
            return  out.squeeze()

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")
# Select initial conditions, separated by intervals of 10MTU 
T = 10
sep = int(T/dt_f)
print(f"Initial conditions separated by {sep} time units")
X_init_conds = X_truth[::sep]
N_init = X_init_conds.shape[0]
nt_total = X_truth.shape[0]

# Check 
nt = int(T/dt_f)
assert(N_init * nt == nt_total)
print(f"Running model for {N_init} initial conditions, for T={T}MTU / {nt} timesteps). Total timesteps={nt_total}. ")

# Run each model for 10MTU
X_all = np.zeros((n_ens, N_init * nt, K))
U_all = np.zeros((n_ens, N_init * nt, K))
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
        X_all[n, i*nt:(i+1)*nt, :] = X
        U_all[n, i*nt:(i+1)*nt, :] = U


# Save results
np.save(f"{save_model_path}X_dtf.npy", X_all)
np.save(f"{save_model_path}U_dtf.npy", U_all)

print(f"Done. Saved to {save_model_path}")