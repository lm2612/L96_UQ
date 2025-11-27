import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pyro
from pyro.infer import Predictive

from ml_models.BayesianModels_old import BayesianLinearRegression, BayesianNN
from utils.summary_stats import summary_stats


from sklearn.decomposition import PCA


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

# Set up directory
data_path = f'./old_data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
plot_path = f'./old_plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'

runtype = 'epistemic_fix'
model_name = 'BayesianNN_hetero_32_N100'
load_model_path = f'./old_data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/{model_name}/' 

# First offline plot
# Get data
X = np.load(f'{data_path}/X_train_dtf.npy')[0:1000]
U = np.load(f'{data_path}/U_train_dtf.npy')[0:1000]

n_ens=100
cmap=plt.cm.jet(np.linspace(0,1,n_ens))


# Now online plot PPE for long climate timescales
# Load PCA
pca = np.load(f"./data/K8_J32_h1_c10_b10_F20/pca_fit.npy", allow_pickle=True).item()
print(pca)


X_truth = np.load(f"{data_path}/truth/X_dtf.npy")

## How often is our simulation in each 'regime' - look for dominant PCs
X_transformed = pca.transform(X_truth)
max_pc = np.argmax(X_transformed, axis=1)
true_regimes = max_pc//2
true_regime_wn1 = np.sum(true_regimes==0)
true_regime_tot = true_regimes.shape[0]
print(true_regimes.shape)
print(true_regime_wn1 / true_regime_tot)

len_time = 200000


filename = f'{data_path}/{model_name}/{runtype}_long_X_dtf.npy' 
X = np.load(filename)[:, :]

n_ens  = X.shape[0]
pred_regimes = np.zeros((n_ens, len_time))

pred_regimes_runtype = []
for m in range(n_ens):
    X_transformed = pca.transform(X[m])
    NT = X.shape[1]
    max_pc = np.argmax(X_transformed, axis=1)
    pred_regimes[ m, :] = max_pc//2

time_inds = range(100, len_time, 100)
time = np.arange(100*dt_f, len_time *dt_f, 100 *dt_f)
percent_spent_in_regime_1 = np.zeros((n_ens, len(time_inds)))
for j, t in enumerate(time_inds):
    percent_spent_in_regime_1[:, j] = pred_regimes[ :, :t].mean(axis=-1)

plt.clf()
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")

mean_percent_spent_in_regime_1 = percent_spent_in_regime_1[ :].mean(axis=0)
ax.plot(time, mean_percent_spent_in_regime_1, 
color="r",
        lw=2, alpha = 0.8)
for n in range(n_ens):
    ax.plot(time, percent_spent_in_regime_1[n], 
        color=cmap[n],
        lw=1, alpha = 0.4)
    
plt.legend()
plt.xlabel("Time (MTU)")
plt.ylabel("Fraction of time spent in regime 1")
plt.savefig(f"{plot_path}/PPE_regime_ens_mem.png")
print(f"{plot_path}/PPE_regime_ens_mem.png")
