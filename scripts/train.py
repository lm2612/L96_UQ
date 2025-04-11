import os
import numpy as np
import matplotlib.pyplot as plt

import torch

from ml_models.TorchModels import LinearRegression, NN

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
dt_f = 0.05
seed = 123
np.random.seed(seed)

N_train = 100
model_name =  f"NN_2layer_N{N_train}"      # Choose LinearRegression or NN 
model = NN(1, 1, [32, 32])
total_params = sum(p.numel() for p in model.parameters())

print("TOTAL PARAMS: ", total_params)

# Set up directory
data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
save_model_path = f'{data_path}/{model_name}/'
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)
print(save_model_path)

# Get data
X = np.load(f'{data_path}/X_train_dtf.npy')
U = np.load(f'{data_path}/U_train_dtf.npy')
print(f'Data loaded from {data_path}')

# Subsample to remove correlations
subsample = 1000 # (1 Time Units)
X = X[::subsample]
U = U[::subsample]

N = X.shape[0]
N_train = N_train #int(0.60 * N)
N_val = max(N - N_train, 0)   # Use remainder for validation

features = np.ravel(X[:N_train])   
targets = np.ravel(U[:N_train])   

features_val = np.ravel(X[N_train:N_train+N_val])   
targets_val = np.ravel(U[N_train:N_train+N_val])    

print(features.shape, targets.shape, features_val.shape, targets_val.shape)

X_torch = torch.tensor(features, dtype=torch.float32).reshape((-1, 1))
Y_torch = torch.tensor(targets, dtype=torch.float32).reshape((-1, 1))

X_val = torch.tensor(features_val, dtype=torch.float32).reshape((-1, 1))
Y_val = torch.tensor(targets_val, dtype=torch.float32).reshape((-1, 1))

# Optimisation settings
optimiser = torch.optim.Adam(params = model.parameters(), lr=1e-2)
loss_function = torch.nn.MSELoss()

num_iterations=500
losses = []
losses_val = []
min_loss = 1E8

for iteration in range(num_iterations):
    model.train()
    optimiser.zero_grad()
    pred = model(X_torch)
    loss = loss_function(pred, Y_torch)
    loss.backward()

    losses.append(loss.item())

    # Update optimiser
    optimiser.step()

    # validation
    model.eval()
    pred = model(X_val)
    loss = loss_function(pred, Y_val)
    losses_val.append(loss.item())

    if loss < min_loss:
        # Save checkpoint
        output_dicts = {
            "iteration": iteration,
            "val_loss": losses_val[-1],
            "train_loss": losses[-1],
            "model": model}

        torch.save(output_dicts, f"{save_model_path}/model_best.pt")
        min_loss = loss

print("Done training")

# Save results
output_dicts = {
    "iteration": iteration,
    "val_loss": losses_val[-1],
    "train_loss": losses[-1],
    "np_rng_state": np.random.get_state(),
    "torch_rng_state": torch.random.get_rng_state(),
    "model":model}

torch.save(output_dicts, f"{save_model_path}/model.pt")
print("Model saved to ", save_model_path)


# Plot and save losses
plt.clf()
figure, ax = plt.subplots(1)
plt.semilogy(losses)
#plt.semilogy(losses_val, alpha=0.5)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.savefig(f"{save_model_path}/losses.png")

# Plot and save result
model.eval()
plt.clf()
figure, ax = plt.subplots(1)
X_domain = torch.linspace(-15, 20., 100).unsqueeze(-1)
pred = model(X_domain).detach()

# Plot
plt.scatter(X_torch.flatten()[::], Y_torch.flatten()[::], color="k", alpha=0.2)
plt.axis(ymin=-15., ymax=20.,xmin=-15., xmax=20.)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel("$X$", fontsize=18)
plt.ylabel("$U$", fontsize=18)
plt.tight_layout()
plt.savefig(f"{save_model_path}/data.png")
plt.plot(X_domain.squeeze(), pred.squeeze(), color="r", linewidth=2)
plt.savefig(f"{save_model_path}/input_outputs_NN.png")
print("Plots done")

