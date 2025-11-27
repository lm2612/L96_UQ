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

# Calculate peak wavenumber
X_fft = np.fft.fft(X_truth, axis=1)
print(X_fft.shape)
k = np.fft.fftfreq(K, d=1/K)
print(k)
X_fft = np.abs(X_fft)
k, X_fft = k[k>0], X_fft[:, k>0]
print(k)
print(X_fft)
peak_ind = np.argmax(X_fft, axis=1)

peak_k = np.array([k[i] for i in peak_ind])
print(peak_k)
plt.plot(peak_k)
plt.show()
print(np.sum(peak_k==1)/peak_k.shape[0], np.sum(peak_k==2)/peak_k.shape[0], np.sum(peak_k==3)/peak_k.shape[0])

