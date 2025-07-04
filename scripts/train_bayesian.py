import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal, AutoLowRankMultivariateNormal, AutoGaussian
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam

from ml_models.BayesianModels import BayesianLinearRegression, BayesianNN
from utils.summary_stats import summary_stats


def bayesian_train(params, training_params, model_name, model, guide):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f,  = params['dt'], params['dt_f']
    N_train = training_params['N_train']
    batch_size, lr, num_iterations = training_params['batch_size'],  training_params['lr'],  training_params['num_iterations']

    
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

    if X_torch.shape[0] > batch_size:
        batch_size = X_torch.shape[0]
    
    dataset = TensorDataset(X_torch, Y_torch)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimisation settings
    adam = pyro.optim.Adam({"lr": lr})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    losses = []
    losses_val = []
    min_loss = 1E8

    pyro.clear_param_store()

    for iteration in range(num_iterations):
        for X_batch, Y_batch in dataloader:
            # calculate the loss and take a gradient step
            loss = svi.step(X_batch, Y_batch)
            losses.append(loss)

            if loss < min_loss:
                # Save checkpoint
                output_dicts = {
                    "iteration": iteration,
                    "train_loss": losses[-1],
                    "model": model,
                    "guide":guide}

                torch.save(output_dicts, f"{save_model_path}/model_best.pt")
                pyro.get_param_store().save( f"{save_model_path}/pyro_best_params.pt")
                min_loss = loss
        
        if iteration % 100 == 0:
            print("[iteration %04d] loss: %.4f" % (iteration + 1, loss ))


    print("Done training")

    # Save results
    output_dicts = {
        "iteration": iteration,
        "train_loss": losses[-1],
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "model":model,
        "guide":guide}

    torch.save(output_dicts, f"{save_model_path}/model.pt")
    pyro.get_param_store().save( f"{save_model_path}/pyro_params.pt")
    print("Model saved to ", save_model_path)


    # Plot and save losses
    plt.clf()
    figure, ax = plt.subplots(1)
    plt.semilogy(losses)
    #plt.semilogy(losses_val, alpha=0.5)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f"{save_model_path}/losses.png")
    print(f"{save_model_path}/losses.png")

    # Plot and save result
    plt.clf()
    figure, ax = plt.subplots(1)
    X_domain = torch.linspace(-15, 20., 100).unsqueeze(-1)
    #pred = model(X_domain).detach()

    # Plot
    plt.scatter(X_torch.flatten()[::], Y_torch.flatten()[::], color="k", alpha=0.2)
    plt.axis(ymin=-15., ymax=20.,xmin=-15., xmax=20.)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.ylabel("$U$", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_model_path}/data.png")


    predictive = Predictive(model, guide=guide, num_samples=800,
                            return_sites=("obs", "_RETURN", "sigma"))
    samples = predictive(X_domain)
    pred_summary = summary_stats(samples)

    plt.plot(X_domain.squeeze(), pred_summary["_RETURN"]["mean"].squeeze(), color="r", linewidth=2)

    plt.fill_between(X_domain.squeeze(), 
                    pred_summary["obs"]["5%"].squeeze(), 
                    pred_summary["obs"]["95%"].squeeze(),
                    color="green", alpha=0.2)

    plt.fill_between(X_domain.squeeze(), 
                    pred_summary["_RETURN"]["5%"].squeeze(), 
                    pred_summary["_RETURN"]["95%"].squeeze(),
                    color="purple", alpha=0.2)

    plt.xlabel("Parameterisation input")
    plt.ylabel("Parameterisation output")
    plt.title("2-layer NN")

    plt.savefig(f"{save_model_path}/input_outputs_NN.png")
    print(f"{save_model_path}/input_outputs_NN.png")

    print("Plots done")


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
    training_params = {'N_train': 50, 
                       'batch_size':128,
                       'N_timesteps':1,
                       'lr': 0.002,
                       'num_iterations' : 10000 ,
}
    N_train = training_params['N_train']
    
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_name =  f"BayesianNN_gaussian_32_N{N_train}"      # Choose LinearRegression or NN 

    # Define model and guide
    model = BayesianNN(1, 1, [32, 32])

    # Guide
    #guide = AutoDiagonalNormal(model)
    guide = AutoLowRankMultivariateNormal(model)
    guide = AutoGaussian(model)

    bayesian_train(params, training_params, model_name, model, guide)
