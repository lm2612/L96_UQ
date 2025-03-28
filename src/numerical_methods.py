## L96 One layer and Two layer models
## Author: Laura Mansfield 
## Date: March 2025
import numpy as np

def Euler_step(X_t, dX_dt, dt, **params):
    """Returns X_t+1 from X_t using Euler's method"""
    return X_t + dX_dt(X_t, **params) * dt

def RK2_step(X_t, dX_dt, dt, **params):
    """Returns X_t+1 from X_t using RK2 method"""
    k1 = dX_dt(X_t, **params)
    k2 = dX_dt(X_t + k1*dt, **params)
    return X_t + (k1 + k2)/2 * dt

def RK4_step(X_t, dX_dt, dt, **params):
    """Returns X_t+1 from X_t using RK4 method"""
    k1 = dX_dt(X_t, **params)
    k2 = dX_dt(X_t + 0.5*k1*dt, **params)
    k3 = dX_dt(X_t + 0.5*k2*dt, **params)
    k4 = dX_dt(X_t + k3*dt, **params)
    return X_t + (k1 + 2*k2 + 2*k3 + k4)/6 * dt

def Euler_step_twolayer(X_t, Y_t, dX_dt, dY_dt, dt, **params):
    """Returns X_t+1, Y_t+1 from X_t, Y_t using Euler's method"""
    X_next, U = X_t + dX_dt(X_t, Y_t, **params) * dt
    Y_next = Y_t + dY_dt(X_next, Y_t, **params) * dt
    return X_next, Y_next, U

def RK4_step_twolayer(X_t, Y_t, dX_dt, dY_dt, dt, **params):
    """Returns X_t+1, Y_t+1 from X_t, Y_t using RK4 method"""
    k1_X, U = dX_dt(X_t, Y_t, **params)
    k1_Y = dY_dt(X_t, Y_t, **params)
    k2_X, _ = dX_dt(X_t + 0.5*k1_X*dt, Y_t + 0.5*k1_Y*dt, **params)
    k2_Y = dY_dt(X_t + 0.5*k1_X*dt, Y_t + 0.5*k1_Y*dt, **params)
    k3_X, _ = dX_dt(X_t + 0.5*k2_X*dt, Y_t + 0.5*k2_Y*dt, **params)
    k3_Y = dY_dt(X_t + 0.5*k2_X*dt, Y_t + 0.5*k2_Y*dt, **params)
    k4_X, _ = dX_dt(X_t + k3_X*dt, Y_t + k3_Y*dt, **params)
    k4_Y = dY_dt(X_t + k3_X*dt, Y_t + k3_Y*dt, **params)
    X_next = X_t + (k1_X + 2*k2_X + 2*k3_X + k4_X)/6 * dt 
    Y_next = Y_t + (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y)/6 * dt
    return X_next, Y_next, U

