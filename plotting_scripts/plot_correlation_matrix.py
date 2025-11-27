import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.utils import parameters_to_vector


import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianNN_Heteroscedastic, BayesianLinearRegression
    
K = 8   
J = 32  

# Define the "true" parameters
h = 1
F = 20  
c = 10
b = 10

# Define time-stepping, random seed
dt = 0.001
dt_f = dt * 5
seed = 123
np.random.seed(seed)

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


# Model name
model_name =  f"BayesianNN_Heteroscedastic_16_16_N100" 
model_path = f"./data/K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/{model_name}/"

# Set up model
output_dicts = torch.load(f"{model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]

scale_tril_constrained = pyro.get_param_store()['AutoMultivariateNormal.scale_tril']

scale_tril = guide.state_dict()['scale_tril_unconstrained']
scale = guide.state_dict()['scale_unconstrained']

cov = scale_tril @ scale_tril.T
cov = scale_tril_constrained @ scale_tril_constrained.T

#plt.imshow(cov.detach(), cmap="RdBu_r", vmax=0.1, vmin=-0.1)
#plt.show()
print(scale.shape, scale_tril.shape)
print(scale)
#plt.imshow(torch.abs(scale_tril), vmax=0.3, vmin=0.)
#plt.show()

#plt.plot(scale)
#plt.show()

# Posterior
print(guide.get_posterior().scale_tril)
#print(guide.)
post_scale = guide.get_posterior().scale_tril
max = torch.max(torch.abs(post_scale))
print(max)
#plt.imshow(scale_tril.detach(), cmap="RdBu_r", vmax=0.1, vmin=-0.1)
#plt.show()


# Compute correlation matrix from  "epistemic" PPE and check its similar
def param_sample(n):
    """ Set up parameterisation for ensemble member n """
    # Open file
    fixed_nn_model = torch.load(f"{model_path}/fixed_param_model_{n}.pt", weights_only=False)
    flat_params = parameters_to_vector(fixed_nn_model.parameters())
    return flat_params

n_ens = 50
params_all = torch.stack([param_sample(n) for n in range(n_ens)])
print(params_all.shape)
params_cov = torch.cov(params_all.T)
def compute_cov(x):
    x_centered = x - x.mean(dim=0)
    cov = (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)
    return cov

params_cov = compute_cov(params_all)
print(params_cov==torch.cov(params_all.T))

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(cov.detach(), cmap="RdBu_r", vmax=0.1, vmin=-0.1)
ax[1].imshow(params_cov.detach(), cmap="RdBu_r", vmax=0.1, vmin=-0.1)
plt.show()

"""fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(cov.detach().sum(dim=0))
ax.plot(params_cov.detach().sum(dim=0))
plt.show()
"""
print(cov+params_cov)

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
parameterisation_AR1 = ParameterisationAR1_Heteroscedastic(pyro_model, guide, include_sigma = False, phi=0.)

def get_params_rand(n):
n_ens = 50
for n in range(n_ens):
    parameterisation_AR1.sample_guide_params()
    print([p for p in parameterisation_AR1.fixed_param_NN.parameters()])
    # Store fixed parameter NN to file
    torch.save(parameterisation_AR1.fixed_param_NN, f"{model_path}/fixed_param_model_{n}.pt")

