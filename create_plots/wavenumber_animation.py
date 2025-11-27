import numpy as np
import matplotlib.pyplot as plt

# Set up parameters for simulation
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

# Gifs
from plotting_scripts.plot_data_gif import plot_data_gif, plot_cartesian
plot_data_gif(params, model_name="", fname_X="X_dtf", 
    fname_Y=None, T=500, save_prefix="presentation_", plot_fn = plot_cartesian)
