## L96 One layer and Two layer models
## Author: Laura Mansfield 
## Date: March 2025
import numpy as np
from numerical_methods import Euler_step, RK2_step, RK4_step, Euler_step_twolayer, RK4_step_twolayer
def dX_dt_onelayer(X_t, F=20, U=0):
    """Returns dX/dt for one layer model"""
    dX_dt = -np.roll(X_t, 1) * (np.roll(X_t, 2) - np.roll(X_t, -1)) -X_t + F + U
    return dX_dt

def dY_dt(X_t, Y_t, F=20, c=10, b=10, h=1.0, J=32, K=8):
    """Returns dY/dt for two layer model"""
    X_int = np.repeat(X_t, J)
    dY_dt = -c*b*np.roll(Y_t, -1) * (np.roll(Y_t, -2) - np.roll(Y_t, 1)) -c*Y_t + (h*c/b) * X_int 
    return dY_dt

def dX_dt_twolayer(X_t, Y_t, F=20, c=10, b=10, h=1.0, J=32, K=8):
    """Returns dX/dt for two layer model"""
    U = - (h*c/b)*Y_t.reshape((K, J)).sum(axis=-1)
    dX_dt = dX_dt_onelayer(X_t, F) + U
    return dX_dt, U

def iterate_onelayer(X_0, dt, T, F=20, K=8):
    """
    Iterates the one layer model
    """
    nt = int(T/dt)
    X = np.zeros((nt, K))
    time = np.zeros(nt)

    X[0] = X_0
    time[0] = 0
    for t in range(1,nt):
        X[t] = RK2_step(X[t-1], dX_dt_onelayer, dt, F=F)
        time[t] = time[t-1] + dt
    return X, time

def iterate_twolayer(X_0, Y_0, dt, T, 
        F=20, c=10, b=10, h=1.0, J=32, K=8):
    """
    Iterates the two layer model
    """
    nt = int(T/dt)
    X = np.zeros((nt, K))
    U = np.zeros((nt, K))
    Y = np.zeros((nt, K*J))

    time = np.zeros(nt)

    Y[0] = Y_0
    X[0] = X_0
    time[0] = 0
    for t in range(1,nt):
        X[t], Y[t], U[t] = RK4_step_twolayer(X[t-1], Y[t-1], dX_dt_twolayer, dY_dt, dt,
                                             F=F, c=c, b=b, h=h, J=J, K=K)
        time[t] = time[t-1] + dt
    return X, Y, U, time

def iterate_onelayer_param(X_0, dt, T, param,
        F=20, c=10, b=10, h=1.0, J=32, K=8):
    """
    Iterates the one layer model with a parameterization of the subgrid-scale forcing
    """
    nt = int(T/dt)
    X = np.zeros((nt, K))
    U = np.zeros((nt, K))

    time = np.zeros(nt)

    X[0] = X_0
    time[0] = 0
    for t in range(1,nt):
        U[t] = param(X[t-1])
        X[t] = RK2_step(X[t-1], dX_dt_onelayer, dt, F=F, U=U[t])
        time[t] = time[t-1] + dt
    return X, U, time