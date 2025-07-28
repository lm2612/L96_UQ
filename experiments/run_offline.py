import os 
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression

from scripts.offline_test import test
from plotting_scripts.plot_ensemble_trajectories import plot_ensembles


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
                'n_ens': 4,
                'N_init': 1,
                'T':1,
                'save_step': 1,
                'F':20                  }

# Model name
model_name =  f"BayesianNN_16_N50" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"
if not os.path.exists(model_path):
    os.makedirs(model_path)
test_params['save_model_path'] = model_path

# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]
predictive = Predictive(pyro_model, guide=guide, num_samples=1, return_sites=("_RETURN", "obs"))

# Run Epistemic with white noise
def param_func(x):
    out = predictive(x.unsqueeze(-1))["obs"]
    return out.squeeze()
test_params['runtype'] = 'both'
test_params['save_prefix'] = 'offline_both_' 
#test(params, test_params, param_func)

run_types = ["offline_both"]
label_names = ["Offline"]
save_prefix = "offline_both_"
plot_ensembles(params, model_name, run_types, label_names, save_prefix=save_prefix, fname="X_dtf", spaghetti=True, shading=True)
exit()



# Run Aleatoric with white noise
fixed_param_NN = pyro_model.get_fixed_param_NN(guide.median())
fixed_param_NN.eval()
def param_func(x):
    with torch.no_grad():
        mean = fixed_param_NN(x.unsqueeze(-1))
        out = pyro_model.sample_obs(mean)
    return out.squeeze()

test_params['runtype'] = 'aleatoric'
test_params['save_prefix'] = 'offline_aleatoric_' 
test(params, test_params, param_func)

# Run both types of uncertainty 
def param_func(x):
    out = predictive(x.unsqueeze(-1))["obs"]
    return out.squeeze()

test_params['runtype'] = 'both'
test_params['save_prefix'] = 'offline_both_' 
#test(params, test_params, param_func)
exit()
fname = "./data/K8_J32_h1_c10_b10_F20/BayesianNN_16_N50//offline_both_X_dtf.npy"
# Load truth data
X_truth = np.load(f"./data/K8_J32_h1_c10_b10_F20/X_dtf.npy")[:5]

# Load ml param model results
X_ml = np.load(fname)

print("TRUTH:" , X_truth)
print("OFFLINE:", X_ml)
# Remove first point if offline
print(" rm ", X_ml[:, 1::2])