import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


import torch
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoMultivariateNormal, AutoLowRankMultivariateNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam

from ml_models.BayesianModels import BayesianLinearRegression, BayesianNN
from utils.summary_stats import summary_stats
from plotting_scripts.plot_data_histogram import plot_hist

def plot_inputs_outputs(params, training_params, model_name, 
    Xmin=-9, Xmax=16, Ymin=-22, Ymax=22, num_samples=800):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f,  = params['dt'], params['dt_f']
    N_train = training_params['N_train']
    training_method = training_params['training_method']
    save_prefix = training_params['save_prefix']

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

    # Plot and save BNN results - first load these from saved file
    if training_method == "mcmc":
        # Assume variational inference
        kernel_name = training_params["kernel_name"]
        output_dicts = torch.load(f"{save_model_path}/{save_prefix}mcmc_{kernel_name}_predictive.pt")
        model = output_dicts["model"]
        predictive = output_dicts["predictive"]
        posterior_samples = output_dicts["samples"]
        training_method = training_method + "_" + kernel_name
        #num_samples = training_params["num_samples"]
        print(f"{training_method}, num samples {num_samples}")
        posterior_samples = OrderedDict( (k, v[:num_samples]) for k, v in posterior_samples.items())

    else:
        # Assume variational inference
        output_dicts = torch.load(f"{save_model_path}/{save_prefix}model_best.pt")
        model = output_dicts["model"]
        guide = output_dicts["guide"]
        pyro.get_param_store().load( f"{save_model_path}/{save_prefix}pyro_best_params.pt")

        # Generate posterior samples for deterministic prediction (mean/median)
        num_samples = 1000
        sample_dicts = [guide() for _ in range(num_samples)]  # list of OrderedDicts

        # Stack each parameter: shape becomes (num_samples, *param_shape)
        posterior_samples = OrderedDict( (k, torch.stack([sd[k] for sd in sample_dicts], dim=0))
            for k in sample_dicts[0].keys()
        )
        # Prepare predictive
        num_samples = 1000
        predictive = Predictive(model, guide=guide, num_samples=num_samples,
                            return_sites=("obs", "_RETURN"))

    # Set up plot
    plt.clf()
    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
        gridspec_kw={'height_ratios': [5, 1], 'hspace':0})
    X_domain = torch.linspace(Xmin, Xmax, 80).unsqueeze(-1)

    # Plot raw data
    plt.sca(ax1)
    plt.scatter(X_torch.flatten()[::], Y_torch.flatten()[::], color="k", alpha=0.2)
    plt.axis(ymin=Ymin, ymax=Ymax, xmin=Xmin, xmax=Xmax)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("$U$", fontsize=18)
    # Add histogram
    plt.sca(ax2)
    plot_hist(X_torch.flatten()[::], ax2)
    plt.axis(ymin=0, xmin=Xmin, xmax=Xmax)
    plt.yticks([], fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{save_model_path}/data.png")
    print(f"{save_model_path}/data.png")

    
    mean_guide = OrderedDict( (k, v.mean(dim=0)) for k, v in posterior_samples.items())
    median_guide = OrderedDict( (k, v.median(dim=0)[0]) for k, v in posterior_samples.items())
    fixed_param_NN = model.get_fixed_param_NN(median_guide)
    median_fixed_param_NN = fixed_param_NN
    median_fixed_param_NN.eval()
    det_pred_median = median_fixed_param_NN(X_domain).detach()
    median_pred = det_pred_median[:, 0]

    plt.sca(ax1)  
    plt.plot(X_domain.squeeze(), median_pred, color="k", linestyle="dashed", linewidth=2, label="median")
    mean_fixed_param_NN = model.get_fixed_param_NN(mean_guide)
    mean_fixed_param_NN.eval()
    det_pred = mean_fixed_param_NN(X_domain).detach()
    mean_pred = det_pred[:, 0]
    plt.plot(X_domain.squeeze(), median_pred, color="k", linewidth=2, label="Mean")
    plt.savefig(f"{save_model_path}/{training_method}_{save_prefix}inputs_outputs_mean.png")

    # Predictive 
    samples = predictive(X_domain)
    pred_summary = summary_stats(samples)

    # Both
    plt.fill_between(X_domain.squeeze(), 
                 pred_summary["obs"]["5%"].squeeze(), 
                 pred_summary["obs"]["95%"].squeeze(),
                 color="dimgrey", alpha=0.2, label="Total")

    # Aleatoric only 
    aleatoric_samples = torch.zeros((num_samples, det_pred.shape[0]))
    for n in range(num_samples):
        aleatoric_samples[n, :] = model.sample_obs(det_pred).detach().squeeze()
    aleatoric_std = torch.std(aleatoric_samples, dim=0)

    plt.fill_between(X_domain.squeeze(), 
                mean_pred - 2*aleatoric_std, 
                mean_pred + 2*aleatoric_std,
                color="seagreen", alpha=0.4, label="Aleatoric")
    
    # Epistemic only
    plt.fill_between(X_domain.squeeze(), 
                 pred_summary["_RETURN"]["5%"][:, 0].squeeze(), 
                 pred_summary["_RETURN"]["95%"][:, 0].squeeze(),
                 color="darkorchid", alpha=0.4, label="Epistemic")
    
    plt.legend()
    plt.axis(ymin=Ymin, ymax=Ymax, xmin=Xmin, xmax=Xmax)

    plt.savefig(f"{save_model_path}/{training_method}_{save_prefix}inputs_outputs_5-95quantiles.png")
    print(f"{save_model_path}/{training_method}_{save_prefix}inputs_outputs_5-95quantiles.png")
    print("Plots done")

    plt.clf()
    # Calculate epistemic and aleatoric and total variances
    samples_mean = samples["_RETURN"][..., 0]
    # Mean prediction is mean of mean
    mean_pred = torch.mean(samples_mean, dim=0)

    # Aleatoric is mean of variance (integrated over all parameters)
    print(samples["_RETURN"].shape, samples["obs"].shape )
    if samples["_RETURN"].shape[-1] > 1:
        samples_sigma2 =  (torch.exp(samples["_RETURN"][..., 1])+model.eps)**2
        aleatoric_var = torch.mean(samples_sigma2, dim=0)
    else:
        # Or for homoscedastic, its a constant
        aleatoric_var =  (pyro.get_param_store()['sigma'].detach())**2
        print(aleatoric_var.shape)
    
    # Epistemic is variance of conditional mean (ignoring aleatoric uncertainty)
    epistemic_var = torch.var(samples_mean, dim=0)
    # Law of total variances
    total_var = aleatoric_var + epistemic_var
    
    # Set up plot
    plt.clf()
    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
        gridspec_kw={'height_ratios': [5, 1], 'hspace':0})

    # Plot raw data
    plt.sca(ax1)
    plt.scatter(X_torch.flatten()[::], Y_torch.flatten()[::], color="k", alpha=0.2)
    plt.axis(ymin=Ymin, ymax=Ymax, xmin=Xmin, xmax=Xmax)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.ylabel("$U$", fontsize=18)
    plt.tight_layout()
    

    # Both
    plt.plot(X_domain.squeeze(), mean_pred, color="k", linewidth=2, label="Mean")
    plt.fill_between(X_domain.squeeze(), 
                 mean_pred - 2*np.sqrt(total_var), 
                 mean_pred + 2*np.sqrt(total_var),
                 color="dimgrey", alpha=0.2, label="Total")
    
    # Aleatoric
    plt.fill_between(X_domain.squeeze(), 
                 mean_pred - 2*np.sqrt(aleatoric_var), 
                 mean_pred + 2*np.sqrt(aleatoric_var),
                 color="seagreen", alpha=0.4, label="Aleatoric")
    # Epistemic
    plt.fill_between(X_domain.squeeze(), 
                 mean_pred - 2*np.sqrt(epistemic_var), 
                 mean_pred + 2*np.sqrt(epistemic_var),
                 color="darkorchid", alpha=0.4, label="Epistemic")

    plt.legend()
    plt.axis(ymin=Ymin, ymax=Ymax, xmin=Xmin, xmax=Xmax)
    # Add histogram
    plt.sca(ax2)
    plot_hist(X_torch.flatten()[::], ax2)
    plt.axis(ymin=0, xmin=Xmin, xmax=Xmax)
    plt.yticks([], fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.tight_layout()

    plt.savefig(f"{save_model_path}/{training_method}_{save_prefix}inputs_outputs.png")
    print(f"{save_model_path}/{training_method}_{save_prefix}inputs_outputs.png")

    # Sanity check how mean prediction (E[Y|X,\theta]) differs from deterministic prediction
    # with mean parameters Y|X,E[\theta]
    plt.clf()
    figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
        gridspec_kw={'height_ratios': [5, 1], 'hspace':0})
    plt.sca(ax1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.ylabel("$\mu$", fontsize=18)
    plt.tight_layout()

    for j in range(num_samples):
            plt.plot(X_domain, samples_mean[j], color="skyblue", lw=0.5, alpha=0.4,
            label = r"$U_{i}|X,\theta$" if j==0 else None)
    plt.plot(X_domain, mean_pred, color="darkorchid", lw=2, label=r"$E[U|X,\theta]$")

    # Compare to deterministic prediction, assuming mean parameter values 
    plt.plot(X_domain, det_pred[:, 0], color="k", lw=2, label=r"$U|X,\bar{\theta}$")

    # Median only different if using MCMC
    #plt.plot(X_domain, det_pred_median[:, 0], color="green", lw=1, linestyle="dashed", 
    #    label="Predicted from median theta")
    plt.legend()
    plt.axis(xmin=Xmin, xmax=Xmax, ymin=Ymin, ymax=Ymax)

    # Add histogram
    plt.sca(ax2)
    plot_hist(X_torch.flatten()[::], ax2)
    plt.axis(ymin=0, xmin=Xmin, xmax=Xmax)
    plt.yticks([], fontsize=18)
    plt.xticks(fontsize=18)
    plt.xlabel("$X$", fontsize=18)
    plt.tight_layout()

    plt.savefig(f"{save_model_path}/{training_method}_{save_prefix}input_aleatoric_mu.png")
    print(f"{save_model_path}/{training_method}_{save_prefix}input_aleatoric_mu.png")

    if samples["_RETURN"].shape[-1] > 1:
        # For heteroscedastic, sanity check how aleatoric mean of variance (E[Var(Y|X,\theta)]) 
        # differs from variance of deterministic prediction with mean parameters Var(Y|X,E[\theta])
        plt.clf()
        figure, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
            gridspec_kw={'height_ratios': [5, 1], 'hspace':0})
        plt.sca(ax1)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("$X$", fontsize=18)
        plt.ylabel("$\sigma$", fontsize=18)
        plt.tight_layout()

        for j in range(num_samples):
            plt.plot(X_domain, np.sqrt(samples_sigma2[j]), color="skyblue", lw=0.5, alpha=0.4,
                        label = r"$\sigma (U_{i}|X,\theta$)" if j==0 else None)

        plt.plot(X_domain, np.sqrt(aleatoric_var), color="seagreen", lw=2, 
            label=r"$(E[\sigma^2(U|X,\theta)])^{1/2}$")

        # Compare to deterministic prediction, assuming mean parameter values 
        aleatoric_predicted_sigma = np.abs(torch.exp(det_pred[:, 1])+model.eps)
        plt.plot(X_domain, aleatoric_predicted_sigma, color="k", lw=2, 
            label=r"$\sigma(U|X,\bar{\theta})$")

        # Median only different if using MCMC
        #aleatoric_predicted_sigma = np.abs(torch.exp(det_pred_median[:, 1])+model.eps)
        #plt.plot(X_domain, aleatoric_predicted_sigma, color="green", lw=1, linestyle="dashed", 
        #    label="Predicted from median theta")

        plt.legend()
        plt.axis( xmin=Xmin, xmax=Xmax, ymin=0, ymax=10.)

        # Add histogram
        plt.sca(ax2)
        plot_hist(X_torch.flatten()[::], ax2)
        plt.axis(ymin=0, xmin=Xmin, xmax=Xmax)
        plt.yticks([], fontsize=18)
        plt.xticks(fontsize=18)
        plt.xlabel("$X$", fontsize=18)
        plt.tight_layout()

        plt.savefig(f"{save_model_path}/{training_method}_{save_prefix}input_aleatoric_sigma.png")
        print(f"{save_model_path}/{training_method}_{save_prefix}input_aleatoric_sigma.png")

    print("Plots done.")
