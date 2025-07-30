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


def plot_inputs_outputs(params, training_params, model_name, 
    Xmin=-14, Xmax=24, Ymin=-22, Ymax=22, num_samples=800):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f,  = params['dt'], params['dt_f']
    N_train = training_params['N_train']
    batch_size, lr, num_iterations = training_params['batch_size'],  training_params['lr'],  training_params['num_iterations']

    # Set up directory
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}'
    save_model_path = f'{data_path}/{model_name}/'

    # Get data
    X = np.load(f'{data_path}/X_train_dtf.npy')
    U = np.load(f'{data_path}/U_train_dtf.npy')
    print(f'Data loaded from {data_path}')

    # Subsample to remove correlations
    subsample = 1000 # (1 Time Units)
    X = X[::subsample]
    U = U[::subsample]

    N = X.shape[0]
    N_train = N_train 
    N_val = max(N - N_train, 0)   # Use remainder for validation

    features = np.ravel(X[:N_train])   
    targets = np.ravel(U[:N_train])   

    features_val = np.ravel(X[N_train:N_train+N_val])   
    targets_val = np.ravel(U[N_train:N_train+N_val])    

    print(features.shape, targets.shape, features_val.shape, targets_val.shape)

    X_torch = torch.tensor(features, dtype=torch.float32).reshape((-1, 1))
    Y_torch = torch.tensor(targets, dtype=torch.float32).reshape((-1, 1))

    # Plot and save best NN so far - first load these from saved file
    output_dicts = torch.load(f"{save_model_path}/model_best.pt")
    model = output_dicts["model"]
    guide = output_dicts["guide"]
    pyro.get_param_store().load( f"{save_model_path}/pyro_best_params.pt")

    # Set up plot
    plt.clf()
    figure, ax = plt.subplots(1)
    X_domain = torch.linspace(Xmin, Xmax, 80).unsqueeze(-1)

    # Plot raw data
    plt.scatter(X_torch.flatten()[::], Y_torch.flatten()[::], color="k", alpha=0.2)
    plt.axis(ymin=Ymin, ymax=Ymax, xmin=Xmin, xmax=Xmax)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.ylabel("$U$", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_model_path}/data.png")
    print(f"{save_model_path}/data.png")

    # Deterministic prediction 
    fixed_param_NN = model.get_fixed_param_NN(guide.median())
    fixed_param_NN.eval()
    det_pred = fixed_param_NN(X_domain).detach()
    mean_pred = det_pred[:, 0]
    plt.plot(X_domain.squeeze(), mean_pred, color="k", linewidth=2, label="mean")
    plt.savefig(f"{save_model_path}/offline_deterministic.png")

    # Predictive 
    num_samples = 800
    predictive = Predictive(model, guide=guide, num_samples=num_samples,
                            return_sites=("obs", "_RETURN"))
    samples = predictive(X_domain)
    pred_summary = summary_stats(samples)

    # Both
    plt.fill_between(X_domain.squeeze(), 
                 pred_summary["obs"]["5%"].squeeze(), 
                 pred_summary["obs"]["95%"].squeeze(),
                 color="dimgrey", alpha=0.2, label="both")

    # Aleatoric only 
    aleatoric_samples = torch.zeros((num_samples, det_pred.shape[0]))
    for n in range(num_samples):
        aleatoric_samples[n, :] = model.sample_obs(det_pred).detach().squeeze()
    std = torch.std(aleatoric_samples, dim=0)
    plt.fill_between(X_domain.squeeze(), 
                mean_pred - 2*std, 
                mean_pred + 2*std,
                color="seagreen", alpha=0.4, label="aleatoric")
    
    # Epistemic only
    plt.fill_between(X_domain.squeeze(), 
                 pred_summary["_RETURN"]["5%"].squeeze(), 
                 pred_summary["_RETURN"]["95%"].squeeze(),
                 color="darkorchid", alpha=0.4, label="epistemic")
    
    plt.legend()
    plt.axis(ymin=Ymin, ymax=Ymax, xmin=Xmin, xmax=Xmax)

    plt.savefig(f"{save_model_path}/input_outputs_NN_2sigma.png")
    print(f"{save_model_path}/input_outputs_NN_2sigma.png")
    print("Plots done")

