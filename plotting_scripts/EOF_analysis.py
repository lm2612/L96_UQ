import os
import numpy as np
import matplotlib.pyplot as plt

import torch
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
print(pca.singular_values_.shape)
print(pca.components_.shape)
print(pca.explained_variance_ratio_.shape)
#plt.plot(np.arange(8),  pca.explained_variance_ratio_)

fig, axs= plt.subplots(n_components//2, figsize=(6, 6))
axs = axs.flatten()
for j in range(n_components):
    axs[j//2].plot(np.arange(K), pca.components_[j], label=f"PCA{j+1} ({100*pca.explained_variance_ratio_[j]:.1f}%)")
    axs[j//2].legend(loc="upper right")
plt.tight_layout()

plot_fname = f"{plot_path}/principal_components_{n_components}.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")


## How often is our simulation in each 'regime' - look for dominant PCs
X_transformed = pca.transform(X_truth)


fig, axs= plt.subplots(n_components//2, figsize=(6, 6), sharex=True)
axs = axs.flatten()
NT = X_truth.shape[0]
for j in range(n_components):
    axs[j//2].plot(np.arange(NT), X_transformed[:, j], label=f"PC{j+1}")
    axs[j//2].legend(loc="upper right")
plt.axis(xmin=0, xmax=1000)
plt.xlabel("Time")
plt.tight_layout()
plot_fname = f"{plot_path}/X_in_principal_component_space.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")

plt.clf()
fig, ax = plt.subplots(1, figsize=(6, 3), sharex=True)
NT = X_truth.shape[0]
for j in range(n_components): 
    ax.plot(np.arange(NT), X_transformed[:, j], label=f"PC{j+1}")
    ax.legend(loc="upper right")
plt.axis(xmin=0, xmax=5000)
plt.xlabel("Time")
plt.tight_layout()
plot_fname = f"{plot_path}/X_in_principal_component_space_all.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")


max_pc = np.argmax(X_transformed, axis=1)
plt.clf()
fig, ax = plt.subplots(1, figsize=(6, 3), sharex=True)
ax.plot(np.arange(NT), max_pc)
plt.axis(xmin=0, xmax=5000)
plt.xlabel("Time")
plt.ylabel("Max PC")
plt.tight_layout()
plot_fname = f"{plot_path}/max_PC_1-4.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")


plt.clf()
fig, ax = plt.subplots(1, figsize=(6, 3), sharex=True)
ax.plot(np.arange(NT), max_pc//2, color="black", label="Truth")
plt.axis(xmin=0, xmax=5000, ymin = -0.5, ymax=1.5)
plt.yticks([0, 1], ["PC1/PC2", "PC3/PC4"])
plt.xlabel("Time")
plt.ylabel("Max PC")
plt.tight_layout()
plot_fname = f"{plot_path}/max_PC_1-2.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")
true_regimes = max_pc//2

runtypes = ['aleatoric', 'epistemic', 'both']
colors = ['seagreen', 'darkorchid', 'blue']
model_name = 'BayesianNN_2layer_N50'
pred_regimes = []
for runtype, color in zip(runtypes,colors):
    filename = f'{data_path}/{model_name}/longrun_{runtype}_X_dtf.npy' 
    X = np.load(filename).squeeze()
    X_transformed = pca.transform(X)
    NT = X.shape[0]
    max_pc = np.argmax(X_transformed, axis=1)
    ax.plot(np.arange(NT), max_pc//2, alpha=0.5, color=color, label=runtype)
    pred_regimes.append(max_pc//2)
model_names = ['NN_2layer_N50','NN_2layer_regime0_N50', 'NN_2layer_regime1_N50' ]   
runtypes.append('deterministic')
runtypes.append('regime1')
runtypes.append('regime2')
for runtype, model_name in zip(runtypes[3:], model_names):
    filename = f'{data_path}/{model_name}/longrun_X_dtf.npy' 
    X = np.load(filename).squeeze()
    X_transformed = pca.transform(X)
    NT = X.shape[0]
    max_pc = np.argmax(X_transformed, axis=1)
    ax.plot(np.arange(NT), max_pc//2, alpha=0.5, color=color, label=runtype)
    pred_regimes.append(max_pc//2)

plt.axis(xmax=1000)
plt.legend()
plot_fname = f"{plot_path}/max_PC_BayesianNN.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")

## Time spent in each regime
plt.clf()
true_regimes = true_regimes[:NT]
print(np.sum(true_regimes==0), np.sum(true_regimes==1))
plt.bar(["Truth"], NT, color="red", label="Regime 2 (wn 1)")
plt.bar(["Truth"], np.sum(true_regimes==0), color="blue", label="Regime 1 (wn 2)")
for i in range(len(runtypes)):
    pred_reg = pred_regimes[i]
    print(runtypes[i], np.sum(pred_reg==0), np.sum(pred_reg==1))
    plt.bar(runtypes[i], NT, color="red")
    plt.bar(runtypes[i], np.sum(pred_reg==0), color="blue")

plt.legend()
plot_fname = f"{plot_path}/time_spent_in_regime.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")


# Time spent in regime for diff F
plt.clf()
true_regimes = true_regimes[:NT]
print(np.sum(true_regimes==0), np.sum(true_regimes==1))
plt.bar(20, NT, color="red", label="Regime 2 (wn 1)")
plt.bar(20, np.sum(true_regimes==0), color="blue", label="Regime 1 (wn 2)")
plt.legend()

Fs=[10, 12, 14, 16, 18, 24, 28, 32, 36]
for F in Fs:
    pert_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    print(pert_path)
    # Load truth data
    X_pert = np.load(f"{pert_path}/X_dtf.npy").squeeze()
    X_transformed = pca.transform(X_pert)
    pert_regimes = np.argmax(X_transformed, axis=1)//2
    plt.bar(F, NT, color="red", label="Regime 2 (wn 1)")
    plt.bar(F, np.sum(pert_regimes==0), color="blue", label="Regime 1 (wn 2)")

plot_fname = f"{plot_path}/time_spent_in_regime_pert.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")