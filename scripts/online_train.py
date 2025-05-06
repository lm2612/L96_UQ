import os
import warnings
import numpy as np
import matplotlib.pyplot as plt

import torch

import pyro
from pyro.infer import Predictive

from ml_models.TorchModels import LinearRegression, NN
from ml_models.BayesianModels import FixedParamNN

from L96.L96_model import L96OneLayerParam

def online_train(params, training_params, model_name, model):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    #T_train = training_params['T_train']     # time (in MTU) over which to train - we will do MSE over this timeframe
    N_t = training_params['N_t']      # Number of timesteps to train over

    # Set up directory
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    save_model_path = f'{data_path}/{model_name}/'
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print(save_model_path)

    # Set up parameterisation for online training
    def param_func(x):
        out = model(x.unsqueeze(-1).clone())
        return out.squeeze()

    # Get data
    X_truth = torch.tensor(np.load(f'{data_path}/X_train_dtf.npy'), dtype=torch.float)
    print(f'Data loaded from {data_path}')

    # X_truth saved at dt_f intervals, but we will run at dt intervals and then compare at dt_f
    subsample_factor = int(dt_f / dt)

    # Select initial conditions, separated by intervals of 10MTU
    T = subsample_factor * dt
    sep = int(T/dt_f)
    print(f"Initial conditions separated by {sep} timesteps")
    #X_init_conds = X_truth[::sep]
    #N_train = X_truth.shape[0]-1
    N_train = N_t
    
    print(f"Training model over {N_train} initial conditions, for T={T}MTU ).  ")
    
    # Optimisation settings
    optimiser = torch.optim.Adam(params = model.parameters(), lr=1e-2)
    loss_function = torch.nn.MSELoss()

    num_iterations=500
    losses = []
    losses_val = []
    min_loss = 1E8
    torch.autograd.set_detect_anomaly(True)
    for i in range(N_train):
        # Initialize model
        l96_model = L96OneLayerParam(X_0=X_truth[i], 
                                    param_func=param_func, 
                                    dt=dt, 
                                    F=F)

        # Run model
        _, _, time = l96_model.iterate(T)
        X_pred = l96_model.X
        loss = loss_function(X_pred, X_truth[i+1])
        loss.backward()

        losses.append(loss.item())

        # Update optimiser
        optimiser.step()

        if i%100==0:
            print(i, loss.item())

        if loss < min_loss:
            # Save checkpoint
            output_dicts = {
                "iteration": i,
                "train_loss": losses[-1],
                "model": model}

            torch.save(output_dicts, f"{save_model_path}/model_best.pt")
            min_loss = loss

    print("Done training")

    # Save results
    output_dicts = {
        "iteration": i,
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
    plt.scatter(X_truth[:-1:10].flatten(), X_truth[1::10].flatten(), color="k", alpha=0.2)
    plt.axis(ymin=-15., ymax=20.,xmin=-15., xmax=20.)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.ylabel("$U$", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_model_path}/data.png")
    plt.plot(X_domain.squeeze(), pred.squeeze(), color="r", linewidth=2)

    plt.xlabel("Parameterisation input")
    plt.ylabel("Parameterisation output")
    plt.title("2-layer NN")
    plt.savefig(f"{save_model_path}/input_outputs_NN.png")
    print("Plots done")
    plt.close()



if __name__ == "__main__":
    params ={
        'F': 20,
        'c': 10,
        'b': 10,
        'h': 1,
        'J': 32,
        'K': 8,
        'dt': 0.001,
        'dt_f': 0.005,
    }
    training_params = {'N_t': 1000}
    N_t = training_params['N_t']
    seeds = range(100, 101)
    for seed in seeds:
        model_name =  f"NN_2layer_online_Nt{N_t}_seed{seed}"      # Choose LinearRegression or NN 
        np.random.seed(seed)
        model = NN(1, 1, [32, 32]) 
        total_params = sum(p.numel() for p in model.parameters())
        online_train(params, training_params, model_name, model)