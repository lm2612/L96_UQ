import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import pickle
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
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
plot_path = f'./plots/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'

# Load truth data
X_truth = np.load(f"{data_path}/truth/X_dtf.npy")

print(X_truth.shape)
n_components=4
pca = PCA(n_components=n_components)
pca.fit(X_truth)
## How often is our simulation in each 'regime' - look for dominant PCs
X_transformed = pca.transform(X_truth)
max_pc = np.argmax(X_transformed, axis=1)
true_regimes = max_pc//2
true_regime_wn1 = np.sum(true_regimes==0)
true_regime_tot = true_regimes.shape[0]
print(true_regimes.shape)
print(true_regime_wn1 / true_regime_tot)

len_time = 200000

runtypes = ['aleatoric_AR1', 'epistemic_AR1', 'both_AR1']
colors = ['seagreen', 'darkorchid', 'dimgrey']
model_name = 'BayesianNN_hetero_32_N100'
pred_regimes = []
n_ens = 50
pred_regimes = np.zeros((len(runtypes), n_ens, len_time))
for r, runtype, color in zip(range(len(runtypes)),runtypes,colors):
    filename = f'{data_path}/{model_name}/{runtype}_long_X_dtf.npy' 
    X = np.load(filename)[:, :]

    #n_ens  = X.shape[0]
    pred_regimes_runtype = []
    for m in range(n_ens):
        X_transformed = pca.transform(X[m])
        NT = X.shape[1]
        max_pc = np.argmax(X_transformed, axis=1)
        pred_regimes[r, m, :] = max_pc//2

time_inds = range(100, len_time, 100)
time = np.arange(100*dt_f, len_time *dt_f, 100 *dt_f)
percent_spent_in_regime_1 = np.zeros((len(runtypes), n_ens, len(time_inds)))
for j, t in enumerate(time_inds):
    percent_spent_in_regime_1[:, :, j] = pred_regimes[:, :, :t].mean(axis=-1)


fig, ax = plt.subplots(1, figsize=(10, 6))
ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
for r in range(len(runtypes)):
    mean_percent_spent_in_regime_1 = percent_spent_in_regime_1[r, :].mean(axis=0)
    ax.plot(time, mean_percent_spent_in_regime_1, 
        color = colors[r], 
        lw=2, alpha = 0.8, 
        label=runtypes[r])
    # Shading
    std_percent_spent_in_regime_1 = percent_spent_in_regime_1[r, :].std(axis=0)
    ax.fill_between(time, mean_percent_spent_in_regime_1 - std_percent_spent_in_regime_1, 
    mean_percent_spent_in_regime_1 + std_percent_spent_in_regime_1, 
        color = colors[r], 
        lw=2, alpha = 0.1)
plt.legend()
plt.xlabel("Time (MTU)")
plt.ylabel("Fraction of time spent in regime 1")
plt.savefig(f"{plot_path}/regime_ens_spread.png")
print(f"{plot_path}/regime_ens_spread.png")

ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
for r in range(len(runtypes)):
    for n in range(n_ens):
        ax.plot(time, percent_spent_in_regime_1[r, n], 
        color = colors[r], 
        lw=1, alpha = 0.3)

plt.savefig(f"{plot_path}/regime_ens_mem.png")
print(f"{plot_path}/regime_ens_mem.png")


# Get mean and std 
time_inds = range(10, len_time, 10)
time = np.arange(10*dt_f, len_time *dt_f, 10 *dt_f)
regime_timeseries_mean = np.zeros((len(runtypes), len(time_inds)))
regime_timeseries_std = np.zeros((len(runtypes), len(time_inds)))
for r in range(len(runtypes)):
    for j, t in enumerate(time_inds):
        mean_ens = pred_regimes[r, :, :t].mean(axis=-1)
        regime_timeseries_mean[r, j] = mean_ens.mean()
        regime_timeseries_std[r, j] =  mean_ens.std()

## Plot

plt.clf()
fig, ax = plt.subplots(1, figsize=(10, 6))
ax.axhline(true_regimes.mean(), color="black", linestyle="dashed", lw=2, label="Truth")
for r in range(len(runtypes)):
    ax.plot(time, regime_timeseries_mean[r], color = colors[r], lw=2, alpha = 0.8, label=runtypes[r])
plt.legend()
plt.xlabel("Time (MTU)")
plt.ylabel("Fraction of time spent in regime 1")
plt.savefig(f"{plot_path}/regime_convergence.png")
print(f"{plot_path}/regime_convergence.png")

plt.clf()
## Plot
fig, ax = plt.subplots(1, figsize=(10, 6))
for r in range(len(runtypes)):
    ax.plot(time, regime_timeseries_std[r], color = colors[r], lw=2, alpha = 0.8, label=runtypes[r])
#ax.axhline(true_regimes.mean(), color="black", linestyle="dashed")
plt.legend()
plt.xlabel("Time (MTU)")
plt.ylabel("Standard deviation across ensemble in fraction of time spent in regime 1")
plt.savefig(f"{plot_path}/regime_uncertainty_time.png")
print(f"{plot_path}/regime_uncertainty_time.png")