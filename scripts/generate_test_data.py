import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from L96.L96_model import L96TwoLayer, subgrid_component

np.random.seed(123)

# Define dimensions of system (fixed)
K = 8   
J = 32  

# Define the "true" parameters
h = 1
F = 20
c = 10
b = 10

# Define time-stepping, random seed
dt = 0.001
T = 1000
seed = 123
np.random.seed(seed)

data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
save_path = f'{data_path}/truth/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Load initial conditions
X0 = np.load(f'{data_path}/X_init.npy')
Y0 = np.load(f'{data_path}/Y_init.npy')

# Set up model
lorenz_model = L96TwoLayer(X_0=X0, Y_0=Y0, F=F, c=c, b=b, h=h, dt=dt)
print(f"Lorenz 1996 model initialized. Running for time={T}...")

# After time = T
X, Y, U_true, time = lorenz_model.iterate(T)
print("Lorenz 1996 model run finished. Saving data...")


## Save test data at dt=0.001
np.save(f'{save_path}/X_full.npy', X)
np.save(f'{save_path}/U_true_full.npy', U_true)

## Save data at dt_f = 0.005 
subsample_factor = 5
dt_f = dt * subsample_factor
print(f"Subsampling data by factor {subsample_factor} to dt_f = {dt_f}")
X = X[::subsample_factor]
U_est = subgrid_component(X[1:], X[:-1], dt_f, F)
np.save(f'{save_path}/U_dtf.npy', U_est)

np.save(f'{save_path}/X_dtf.npy', X)


print(f"Saved to {data_path}/")


