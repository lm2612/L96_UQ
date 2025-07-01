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
spinup_T = 1000
seed = 123
np.random.seed(seed)

data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# Random initial conditions
X0 = np.random.rand(K)
Y0 = np.random.rand(K*J)
# Set up model
lorenz_model = L96TwoLayer(X_0=X0, Y_0=Y0, F=F, c=c, b=b, h=h, dt=dt)
print(f"Lorenz 1996 model initialized. Spin-up {spinup_T}")
X, Y, U_true, time = lorenz_model.iterate(spinup_T)

print(f"Spin-up done. Running for time={T}...")
X, Y, U_true, time = lorenz_model.iterate(T)
print("Lorenz 1996 model run finished. Saving data...")

## Save full data at dt=0.001
np.save(f'{data_path}/X_full.npy', X)
np.save(f'{data_path}/U_true_full.npy', U_true)

# Save last X and Y for initialization of test simulations
np.save(f'{data_path}/X_init.npy', X[-1])
np.save(f'{data_path}/Y_init.npy', Y[-1])

## Save data at dt_f = 0.005 for training
subsample_factor = 5
dt_f = dt * subsample_factor
print(f"Subsampling data by factor {subsample_factor} to dt_f = {dt_f}")
X_all = X
X = X[::subsample_factor]
time_subsample = time[::subsample_factor]
U_est = subgrid_component(X[1:], X[:-1], dt_f, F)
np.save(f'{data_path}/U_train_dtf.npy', U_est)

X = X[1:]
np.save(f'{data_path}/X_train_dtf.npy', X)

plt.clf()
maxtime = 1000
plt.plot(time[:int(maxtime*subsample_factor)], U_true[:int(maxtime*subsample_factor),0], label="True U")
plt.plot(time_subsample[:maxtime], U_est[:maxtime,0], label="Estimated U")
plt.legend()
plt.ylabel("U")
plt.xlabel("Time")
plt.savefig(f"{data_path}/time_v_U.png")

U_true_all = U_true
U_true = U_true[::subsample_factor][1:]
np.save(f'{data_path}/U_true_dtf.npy', U_true)

print(f"Saved to {data_path}/")

plt.clf()
plt.scatter(U_true, U_est, alpha=0.2)
plt.savefig(f"{data_path}/U_true_v_U_est.png")

plt.clf()
plt.scatter(X_all, U_true_all, alpha=0.1, color="k", label="Exact U")
plt.scatter(X, U_est, alpha=0.1, color="b", label="Est U")
plt.legend()
plt.savefig(f"{data_path}/X_v_Utrue_Uest.png")


