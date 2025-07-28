import os 
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression


from scripts.online_test import test

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

n_ens = 50
N_init = 20
# Model name
model_name =  f"BayesianNN_16_N50" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]

concat_files_flag = True
def concat_files(model_path, save_prefix, n_ens, fnames=['X_dtf', 'U_dtf', 'test_params']):
    """ Put all files back into one file for easier reading and comparison """
    for fname in fnames:
        paths = [model_path+f"IC{m:02d}_{save_prefix}{fname}.npy" for m in range(n_ens)]
        X = [np.load(path, allow_pickle=True) for path in paths]
        X = np.stack(X, axis=0).squeeze()
        np.save(f"{model_path}/IC_{save_prefix}{fname}.npy", X)
        [os.remove(path) for path in paths]
        print(f"Individual files removed. All saved to {model_path}/IC_{save_prefix}{fname}.npy")
        

# Deterministic
save_prefix = 'deterministic_'
runtype = 'deterministic'
# Deterministic - no uncertainty
fixed_param_NN = pyro_model.get_fixed_param_NN(guide.median())
def param_func(x):
    with torch.no_grad():
        mean = fixed_param_NN(x.unsqueeze(-1))
    return mean.squeeze()

for m in range(n_ens):
    test_params = { 'fname':f'IC{m:02d}_X_dtf.npy',
                    'runtype': None,
                    'save_model_path':model_path,
                    'save_prefix':f'IC{m:02d}_{save_prefix}',
                    'runtype':runtype,
                    'n_ens': 1,
                    'N_init': N_init,
                    'save_step': 1,
                    'T':10 ,
                    'F':20                  }
    test(params, test_params, param_func)

if concat_files_flag:
    concat_files(model_path, save_prefix, n_ens, fnames=['X_dtf', 'U_dtf', 'test_params'])
    
# Repeat for epistemic , aleatoric, etc. - for these we run just 1 ens member for each IC to sample both IC and epi/ale
save_prefix = 'epistemic_'
runtype = 'epistemic'
# Run Epistemic with white noise - sample guide each time
def param_func(x):
    fixed_param_NN = pyro_model.get_fixed_param_NN(guide())
    fixed_param_NN.eval()
    with torch.no_grad():
        out = fixed_param_NN(x.unsqueeze(-1))
    return out.squeeze()
    
for m in range(n_ens):
    test_params = { 'fname':f'IC{m:02d}_X_dtf.npy',
                    'runtype': None,
                    'save_model_path':model_path,
                    'save_prefix':f'IC{m:02d}_{save_prefix}',
                    'runtype':runtype,
                    'n_ens': 1,
                    'N_init': N_init,
                    'save_step': 1,
                    'T':10 ,
                    'F':20                  }
    test(params, test_params, param_func)

if concat_files_flag:
    concat_files(model_path, save_prefix, n_ens, fnames=['X_dtf', 'U_dtf', 'test_params'])
        
# Run Aleatoric with white noise
save_prefix = 'aleatoric_'
runtype = 'aleatoric'
fixed_param_NN = pyro_model.get_fixed_param_NN(guide.median())
fixed_param_NN.eval()
def param_func(x):
    with torch.no_grad():
        mean = fixed_param_NN(x.unsqueeze(-1))
        out = pyro_model.sample_obs(mean)
    return out.squeeze()

for m in range(n_ens):
    test_params = { 'fname':f'IC{m:02d}_X_dtf.npy',
                    'runtype': None,
                    'save_model_path':model_path,
                    'save_prefix':f'IC{m:02d}_{save_prefix}',
                    'runtype':runtype,
                    'n_ens': 1,
                    'N_init': N_init,
                    'save_step': 1,
                    'T':10 ,
                    'F':20                  }
    test(params, test_params, param_func)


if concat_files_flag:
    concat_files(model_path, save_prefix, n_ens, fnames=['X_dtf', 'U_dtf', 'test_params'])

# Run both types of uncertainty 
save_prefix = 'both_'
runtype = 'both'
def param_func(x):
    fixed_param_NN = pyro_model.get_fixed_param_NN(guide())
    fixed_param_NN.eval()
    with torch.no_grad():
        mean = fixed_param_NN(x.unsqueeze(-1))
        out = pyro_model.sample_obs(mean)
    return out.squeeze()

for m in range(n_ens):
    test_params = { 'fname':f'IC{m:02d}_X_dtf.npy',
                    'runtype': None,
                    'save_model_path':model_path,
                    'save_prefix':f'IC{m:02d}_{save_prefix}',
                    'runtype':runtype,
                    'n_ens': 1,
                    'N_init': N_init,
                    'save_step': 1,
                    'T':10 ,
                    'F':20                  }
    test(params, test_params, param_func)

if concat_files_flag:
    concat_files(model_path, save_prefix, n_ens, fnames=['X_dtf', 'U_dtf', 'test_params'])
