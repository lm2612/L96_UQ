import numpy as np
import os

def concat_files(model_path, save_prefix, n_ens, fnames=['X_dtf', 'U_dtf', 'test_params']):
    """ Put all files back into one file for easier reading and comparison """
    for fname in fnames:
        paths = [model_path+f"IC{m:02d}_{save_prefix}{fname}.npy" for m in range(n_ens)]
        X = [np.load(path, allow_pickle=True) for path in paths]
        X = np.stack(X, axis=0).squeeze()
        np.save(f"{model_path}/IC_{save_prefix}{fname}.npy", X)
        [os.remove(path) for path in paths]
        print(f"Individual files removed. All saved to {model_path}/IC_{save_prefix}{fname}.npy")
        