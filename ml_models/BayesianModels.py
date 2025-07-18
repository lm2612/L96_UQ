import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

from ml_models.TorchModels import LinearRegression, NN

class BayesianLinearRegression(PyroModule):
    """Bayesian linear regression"""
    def __init__(self, n_features=1, n_targets=1):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets

        self.linear = PyroModule[torch.nn.Linear](n_features, n_targets)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([n_targets, n_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([n_targets]).to_event(1))

    def forward(self, X, Y=None):
        sigma = pyro.sample("sigma", dist.Uniform(1.0e-6, 10.))
        mean = self.linear(X) 
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=Y)
        return mean

    def get_fixed_param(self, guide):
        """Returns a NN torch module in same format as this model but with parameters fixed based on guide 
        dictionary"""
        return LinearRegression(self.n_features, self.n_targets, param_dict = param_dict)

    def sample_obs(self, mean):
        """Sample aleatoric noise (e.g., if using deterministic pred w/ fixed parameters)"""
        sigma = pyro.param("sigma", dist.Uniform(1.0e-6, 10.))
        with pyro.plate("data", mean.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1))
        return obs

class BayesianNN(PyroModule):
    """Bayesian neural network with arbitrary number of hidden layers - can be used for epistemic uncertainty (_RETURN) or both
    aleatoric and epistemic (obs)"""
    def __init__(self, n_features=1, n_targets=1, n_hidden=[16]):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_hidden = n_hidden

        nodes = [n_features]+n_hidden+[n_targets]

        self.layers = PyroModule[torch.nn.ModuleList]([])
        for j in range(len(nodes)-1):
            linear_j = PyroModule[torch.nn.Linear](nodes[j], nodes[j+1])
            linear_j.weight = PyroSample(dist.Normal(0., 1.).expand([nodes[j+1], nodes[j]]).to_event(2))
            linear_j.bias = PyroSample(dist.Normal(0., 10.).expand([nodes[j+1]]).to_event(1))
            self.layers.append(linear_j)

        self.activation_function = torch.nn.ReLU()

    def forward(self, X, Y=None):
        for j in range(len(self.layers)-1):
            X = self.layers[j](X)
            X = self.activation_function(X)
        mean = self.layers[-1](X)

        sigma = pyro.param("sigma", dist.Uniform(1.0e-6, 10.))
        with pyro.plate("data", X.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=Y)
        return mean

    def get_fixed_param_NN(self, param_dict):
        """Returns a NN torch module in same format as this model but with parameters fixed based on guide 
        dictionary"""
        return NN(self.n_features, self.n_targets, self.n_hidden, param_dict = param_dict)

    def sample_obs(self, mean):
        """Sample aleatoric noise (e.g., if using deterministic pred w/ fixed parameters)"""
        sigma = pyro.param("sigma", dist.Uniform(1.0e-6, 10.))
        with pyro.plate("data", mean.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1))
        return obs
