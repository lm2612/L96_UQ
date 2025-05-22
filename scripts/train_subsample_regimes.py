import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.decomposition import PCA

from ml_models.TorchModels import LinearRegression, NN, NNDropout


def train(params, training_params, model_name, model):
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    N_train = training_params['N_train']
    batch_size = training_params['batch_size']
    regime_number = training_params['regime']

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

    ## Calculate regimes and subsample based on that
    # Load truth data
    X_truth = np.load(f"{data_path}/truth/X_dtf.npy")
    print(X_truth.shape)
    n_components=4
    pca = PCA(n_components=n_components)
    pca.fit(X_truth)
    X_transformed = pca.transform(X)
    regime = np.argmax(X_transformed, axis=1)//2

    regime_0_inds = np.argwhere(regime == regime_number)
    regime_1_inds = np.argwhere(regime == 1 - regime_number)
    features = np.ravel(X[regime_0_inds])   
    targets = np.ravel(U[regime_0_inds])   

    features_val = np.ravel(X[regime_1_inds])   
    targets_val = np.ravel(U[regime_1_inds])    

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
    plt.scatter(X_val.flatten()[::], Y_val.flatten()[::], color="b", alpha=0.2)
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
    training_params = {'N_train': 50, 
                       'batch_size':128,
                       'N_timesteps':1}
    N_train = training_params['N_train']
    regimes = [0, 1]
    seed = 100
    for regime in regimes:
        training_params['regime'] = regime
        model_name =  f"NN_2layer_regime{regime}_N{N_train}"      # Choose LinearRegression or NN 
        np.random.seed(seed)
        model = NN(1, 1, [32, 32]) #, dropout_rate=0.5)
        total_params = sum(p.numel() for p in model.parameters())
        print("TOTAL PARAMS: ", total_params)
        train(params, training_params, model_name, model)
