import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from L96.L96_model import L96TwoLayer, subgrid_component

def generate_truth(params, test_params):
    """Function that does online test and saves output
    Args:
    - params
    """
    K, J, h, F, c, b = params['K'], params['J'], params['h'], params['F'], params['c'], params['b']
    dt, dt_f = params['dt'], params['dt_f']
    # Get test params
    save_path = test_params['save_path']
    T = test_params['T']
    # Optional load/save prefix for file names, e.g., for very long run split into 00, 01, 02, ...
    load_prefix, save_prefix  = test_params['load_prefix'], test_params['save_prefix'] 
    # Only save all Ys if set to True, otherwise just save last point for start of ICs
    save_Y, save_ICs = test_params['save_Y'],  test_params['save_ICs']
    
    
    data_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
    # Load initial conditions
    X0 = np.load(f'{data_path}/{load_prefix}X_init.npy') 
    Y0 = np.load(f'{data_path}/{load_prefix}Y_init.npy') 

    # May wish to change F to run with different forcing 
    F = test_params['F']

    save_path = f'./data/K{K}_J{J}_h{h}_c{c}_b{b}_F{F}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Set up model
    lorenz_model = L96TwoLayer(X_0=X0, Y_0=Y0, F=F, c=c, b=b, h=h, dt=dt)
    print(f"Lorenz 1996 model initialized. Running for time={T}...")

    # After time = T
    X, Y, U_true, time = lorenz_model.iterate(T)
    print("Lorenz 1996 model run finished. Saving data...")


    ## Save test data at dt=0.001
    np.save(f'{save_path}/{save_prefix}X_full.npy', X)
    np.save(f'{save_path}/{save_prefix}U_true_full.npy', U_true)

    ## Save data at dt_f = 0.005 
    subsample_factor = int(dt_f / dt)
    print(f"Subsampling data by factor {subsample_factor} to dt_f = {dt_f}")
    X = X[::subsample_factor]
    Y = Y[::subsample_factor]
    U_est = subgrid_component(X[1:], X[:-1], dt_f, F)
    np.save(f'{save_path}/{save_prefix}U_dtf.npy', U_est)

    np.save(f'{save_path}/{save_prefix}X_dtf.npy', X)
    if save_Y:
        np.save(f'{save_path}/{save_prefix}Y_dtf.npy', Y)

    # Save last point in case you want to restart run 
    if save_ICs:
        np.save(f'{data_path}/{save_prefix}X_init.npy', X[-1]) 
        np.save(f'{data_path}/{save_prefix}Y_init.npy', Y[-1]) 
    print(f"Saved to {save_path}/{save_prefix}")



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
    test_params = { 'save_path':f"K{params['K']}_J{params['J']}_h{params['h']}_c{params['c']}_b{params['b']}_F{params['F']}/",
                    'load_prefix': '',
                    'save_prefix': '', 
                    'save_Y': True,
                    'save_ICs': False,
                    'T':1000 ,
                    'F':20                  }

    
    seed = 123
    np.random.seed(seed)
    generate_truth(params, test_params)

