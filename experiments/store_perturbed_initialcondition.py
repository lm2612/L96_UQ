import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import BayesianNN, BayesianLinearRegression


from scripts.online_test import test


K = 8
seed = 123
# Open and save different initial condition files to create IC uncertainty
X_truth = np.load(f"./data/K8_J32_h1_c10_b10_F20/X_dtf.npy")
# Multiply by random numbers sampled across (1e-16)
n_ens = 50
eps = 1e-8
print(X_truth.shape)

rng = np.random.default_rng(seed)
for m in range(n_ens):
    X_perturbed = X_truth + 2*(rng.random(X_truth.shape)-0.5)*eps
    np.save(f"./data/K8_J32_h1_c10_b10_F20/IC{m:02d}_X_dtf.npy", X_perturbed)
    print(f"Saved as ./data/K8_J32_h1_c10_b10_F20/IC{m:02d}_X_dtf.npy")




