import xarray as xr
import numpy as np

from scipy.stats import gaussian_kde


# Kde plot
def kde_plot(X, X_domain=np.linspace(-25, 25, 80),  bw=0.15):
    if type(X)==xr.core.dataarray.DataArray:
        X = X.to_numpy()
    X = X.flatten()
    kde = gaussian_kde(X, bw_method=bw)
    pdf = kde.pdf(X_domain)
    return(pdf)
