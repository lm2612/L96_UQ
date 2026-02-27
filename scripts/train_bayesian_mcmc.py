import os
import numpy as np
import matplotlib.pyplot as plt

#import dill
import torch
from torch.utils.data import TensorDataset, DataLoader

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.nn import PyroModule, PyroSample
from pyro.infer import MCMC, NUTS, HMC, Trace_ELBO, Predictive, RandomWalkKernel
from pyro.optim import Adam

from ml_models.BayesianModels import BayesianLinearRegression, BayesianNN, BayesianNN_Heteroscedastic
from utils.summary_stats import summary_stats
from utils.param_sample import param_sample
from plotting_scripts.plot_inputs_outputs import plot_inputs_outputs

def bayesian_train_mcmc(params, training_params, model_name, model, kernel_name="NUTS"):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f,  = params['dt'], params['dt_f']
    N_train = training_params['N_train']
    num_samples, warmup_steps = training_params['num_samples'],  training_params['warmup_steps']
    num_chains = training_params['num_chains']

    
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
    N_train = N_train 
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

    pyro.clear_param_store()

    # Optimisation settings: MCMC
    # Define MCMC sampler
    if kernel_name == "NUTS":
        # NUTS = "No-U-Turn Sampler" (https://arxiv.org/abs/1111.4246), gives HMC an adaptive step size
        kernel = NUTS(model, jit_compile=False)
    elif kernel_name == "HMC":
        # Hamiltonian Monte Carlo sampler
        kernel = HMC(model)
    elif kernel_name == "RW":
        kernel = RandomWalkKernel(model)

    mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, 
        num_chains=num_chains)
    
    mcmc.run(X_torch, Y_torch)
    samples = mcmc.get_samples()
    print("Done")

    # Inference
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples(), 
                            return_sites=("obs", "_RETURN"))
    
    # Save results - directly save mcmc samples
    output_dicts = {
        "iteration": warmup_steps+num_samples,
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "model":model,
        "predictive":predictive,
        "samples":samples
        }

    torch.save(output_dicts, f"{save_model_path}/mcmc_{kernel_name}_predictive.pt")
    print("Model saved to ", save_model_path)

    ## Plot log likelihood with iteration
    log_like = []
    for it in range(num_samples):
        params = param_sample(it, samples)
        conditioned_model = poutine.condition(model, params)
        trace = poutine.trace(conditioned_model).get_trace(X=X_torch, Y=Y_torch)
        log_like.append(trace.log_prob_sum().item())
        
    # Plot likelihood 
    # Set up plot
    plt.clf()
    figure, ax = plt.subplots(1)
    plt.plot(range(num_samples), log_like)
    plt.xlabel("Iterations")
    plt.ylabel("Log likelihood")
    plt.savefig(f"{save_model_path}/loglikelihood_{kernel_name}.png")
    print(f"{save_model_path}/loglikelihood_{kernel_name}.png")

    # Plot inputs and outputs for best NN saved
    plot_inputs_outputs(params, training_params, model_name)


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
    training_params = {'N_train': 100, 
                       'N_timesteps':1,
                       'num_samples' : 500 ,
                       'warmup_steps' : 200 ,
}
    N_train = training_params['N_train']
    
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_name =  f"BayesianNN_Heteroscedastic_16_16_N{N_train}_priorNormal_5"      # Choose LinearRegression or NN 

    # Define model and guide
    model = BayesianNN_Heteroscedastic(1, 1, [16, 16])

    bayesian_train_mcmc(params, training_params, model_name, model, kernel_name="RW")
    
