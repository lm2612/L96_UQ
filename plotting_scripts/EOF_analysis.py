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
plot_path = f'{data_path}/'

# Load truth data
X_truth = np.load(f"{data_path}/X_dtf.npy")

n_components=4
pca = PCA(n_components=n_components)
pca.fit(X_truth)

# Save PCA object
#np.save(f"{data_path}/pca_fit.npy", pca)
print(f"saved as {data_path}/pca_fit.npy")

print(pca.singular_values_.shape)
print(pca.components_.shape)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])
print(pca.explained_variance_ratio_[2]+pca.explained_variance_ratio_[3])
print(pca.explained_variance_ratio_.sum())
print((pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1])/pca.explained_variance_ratio_.sum())
print((pca.explained_variance_ratio_[2]+pca.explained_variance_ratio_[3])/pca.explained_variance_ratio_.sum())
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


max_time = 20000
truth_regimes = max_pc//2

plt.clf()
fig, ax = plt.subplots(1, figsize=(8, 3), sharex=True)
ax.plot(np.arange(NT), truth_regimes, color="black", label="Truth")
plt.axis(xmin=0, xmax=5000, ymin = -0.5, ymax=1.5)
plt.yticks([0, 1], ["k=2", "k=1"])
plt.xlabel("Time")
plt.ylabel("Max PC")
plt.tight_layout()
plot_fname = f"{plot_path}/max_PC_1-2.png"
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")
plt.axis(xmin=0, xmax=500, ymin = -0.5, ymax=1.5)
plot_fname = f"{plot_path}/wavenumbers.png"
plt.ylabel("Regime")
plt.savefig(plot_fname)
print(f"Saved to {plot_fname}")
 


truth_regimes = max_pc//2
true_regime_wn1 = np.sum(truth_regimes==0)
print(truth_regimes.shape)
true_regime_tot = truth_regimes.shape[0]
print(true_regime_wn1 / true_regime_tot)
print(np.sum(truth_regimes==1) / true_regime_tot)
print(truth_regimes.mean())
