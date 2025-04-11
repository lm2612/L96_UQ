import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
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

N_train = 100
model_name = f"NN_2layer_N{N_train}"      # Choose LinearRegression or NN  or OneLayer
model_name = f"LinearRegression_N{N_train}"
runtype = "aleatoric"    # epistemic, aleatoric or None
n_ens = 20               # number of times to run for (for deterministic this will be 1)
# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_path = f'{data_path}/{model_name}/'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
print(save_model_path)


# Load ml param model
if "Bayesian" in model_name:
    print(f"Running a stochastic parameterisation: {model_name} using {runtype} uncertainty, {n_ens} ensemble members")
    # Stochastic parameterisation
    if runtype == "epistemic":
        return_site = "_RETURN" 
    elif runtype == "aleatoric":
        return_site = "obs"
    elif runtype == "mean":
        # Todo
        return_site = "obs"
    else:
        raise ValueError(f"{runtype} unknown, must be epistemic, aleatoric or mean.")

    output_dicts = torch.load(f"{save_model_path}/model_best.pt")
    pyro.get_param_store().load(f"{save_model_path}/pyro_params.pt")

    pyro_model = output_dicts["model"]
    guide = output_dicts["guide"]
    predictive = Predictive(pyro_model, guide=guide, num_samples=1,
                        return_sites=(return_site,))

    def param_func(x):
        nn_input = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        out = predictive(nn_input)[return_site]
        return out.squeeze().numpy()

    save_model_path = f'{save_model_path}/{runtype}_'

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
        def param_func(X):
            return  np.zeros_like(X)
    else:
        output_dicts = torch.load(f"{save_model_path}/model_best.pt")
        ml_model = output_dicts["model"]
        ml_model.eval()

        # Initialize param_func
        def param_func(X):
            nn_input = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
            with torch.no_grad():
                out = ml_model(nn_input)
            return  out.squeeze().numpy()

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

