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
K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']

T = 10 
AC1 = 0.984865        

N_train = 50
model_name =  f"BayesianNN_2layer_N{N_train}" 
runtype = 'aleatoric'

# Set up directories
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/truth/'
load_model_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/{model_name}/' 

output_dicts = torch.load(f"{load_model_path}/model_best.pt", weights_only=False)
pyro.get_param_store().load(f"{load_model_path}/pyro_params.pt")
pyro_model = output_dicts["model"]
guide = output_dicts["guide"]

return_site = "obs"
fixed_param_NN = FixedParamNN(pyro_model, guide)
fixed_param_NN.eval()
predictive = Predictive(fixed_param_NN, guide=guide, num_samples=1,
            return_sites=(return_site,))
X = torch.Tensor([2.9342])
print("Pyro sampling")
samples = np.concatenate([ predictive(X)['obs'].numpy() for j in range(1000)]).squeeze()
print(samples)
plt.hist(samples, bins=50, alpha=0.5)
#plt.show()

return_site = "_RETURN"
fixed_param_NN = FixedParamNN(pyro_model, guide)
fixed_param_NN.eval()
predictive = Predictive(fixed_param_NN, guide=guide, num_samples=1,
            return_sites=(return_site,))

print("My sampling")
sigma = guide.median()['sigma'].numpy()
print(sigma)

my_samples = np.concatenate([ predictive(X)['_RETURN'].numpy() + sigma*np.random.randn(X.shape[0]) for j in range(1000)]).squeeze()
print(my_samples)
plt.hist(my_samples, bins=50, alpha=0.5)
plt.show()

phi = 0.984865
sigma_e2 =  4.580670356750488
sigma_e = np.sqrt(sigma_e2)
# Get AR1 noise
stochastic_err = []
e = 0.
plt.clf()
for t in range(100):
    e = phi * e + np.sqrt(1-phi**2) * sigma * np.random.randn(1)
    stochastic_err.append(e)
plt.plot(stochastic_err)
plt.show()