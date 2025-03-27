## L96 One layer and Two layer models
## Author: Laura Mansfield 
## Date: March 2025
import numpy as np

def dX_dt_1(X_t, F):
    """Returns dX/dt for one layer model"""
    dX_dt = -np.roll(X_t, 1) * (np.roll(X_t, 2) - np.roll(X_t, -1)) -X_t + F
    return dX_dt

def dY_dt(X_t, Y_t, c, b, h):
    """Returns dY/dt for two layer model"""
    X_int = np.repeat(X_t, J)
    dY_dt = -c*b*np.roll(Y_t, -1) * (np.roll(Y_t, -2) - np.roll(Y_t, 1)) -c*Y_t + (h*c/b) * X_int 
    return dY_dt


