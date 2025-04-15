import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class BayesianLinearRegression(PyroModule):
    """Bayesian linear regression"""
    def __init__(self, n_features=1, n_targets=1):
        super().__init__()
        self.linear = PyroModule[torch.nn.Linear](n_features, n_targets)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([n_targets, n_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([n_targets]).to_event(1))

    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(1.0e-6, 10.))
        mean = self.linear(x) 
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
        return mean

class BayesianNN(PyroModule):
    """Bayesian neural network with two hidden layers"""
    def __init__(self, n_features=1, n_targets=1, n_hidden=[32, 32]):
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_hidden = n_hidden

        # 2 layers NN
        self.linear_1 = PyroModule[torch.nn.Linear](n_features, n_hidden[0])
        self.linear_1.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden[0], n_features]).to_event(2))
        self.linear_1.bias = PyroSample(dist.Normal(0., 10.).expand([n_hidden[0]]).to_event(1))

        self.linear_2 = PyroModule[torch.nn.Linear](n_hidden[0], n_hidden[1])
        self.linear_2.weight = PyroSample(dist.Normal(0., 1.).expand([n_hidden[1], n_hidden[0]]).to_event(2))
        self.linear_2.bias = PyroSample(dist.Normal(0., 10.).expand([n_hidden[1]]).to_event(1))

        self.linear_3 = PyroModule[torch.nn.Linear](n_hidden[1], n_targets)
        self.linear_3.weight = PyroSample(dist.Normal(0., 1.).expand([n_targets, n_hidden[1]]).to_event(2))
        self.linear_3.bias = PyroSample(dist.Normal(0., 10.).expand([n_targets]).to_event(1))

        self.activation_function = torch.nn.ReLU()


    def forward(self, x, y=None):
        sigma = pyro.sample("sigma", dist.Uniform(1.0e-6, 10.))

        mean = self.linear_1(x)
        mean = self.activation_function(mean)
        mean = self.linear_2(mean)
        mean = self.activation_function(mean)
        mean = self.linear_3(mean)

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
        return mean
