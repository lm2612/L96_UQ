import numpy as np 
import matplotlib.pyplot as plt


def plot_hist(X, ax=None):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    plt.sca(ax)
    plt.hist(X, bins=30, color="dimgrey", alpha=0.5)
    return ax
