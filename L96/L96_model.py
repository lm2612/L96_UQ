## L96 One layer and Two layer models
## Author: Laura Mansfield 
## Date: March 2025
import numpy as np
import torch
from L96.numerical_methods import Euler_step, RK2_step, RK4_step, Euler_step_twolayer, RK4_step_twolayer

# First define generic functions for the one layer and two layer models
def dX_dt_onelayer(X_t, F=20, U=0):
    """Returns dX/dt for one layer model with optional subgrid-scale forcing term U"""
    dX_dt = -torch.roll(X_t, 1, dims=-1) * (torch.roll(X_t, 2, dims=-1) - torch.roll(X_t, -1, dims=  -1)) - X_t + F + U
    return dX_dt

def dY_dt(X_t, Y_t, F=20, c=10, b=10, h=1.0, J=32, K=8):
    """Returns dY/dt for two layer model"""
    X_int = torch.repeat_interleave(X_t, J)
    dY_dt = -c*b*torch.roll(Y_t, -1, dims=-1) * (torch.roll(Y_t, -2, dims=-1) - torch.roll(Y_t, 1, dims=-1)) -c*Y_t + (h*c/b) * X_int 
    return dY_dt

def dX_dt_twolayer(X_t, Y_t, F=20, c=10, b=10, h=1.0, J=32, K=8):
    """Returns dX/dt for two layer model"""
    U = - (h*c/b)*Y_t.reshape((K, J)).sum(dim=-1)
    dX_dt = dX_dt_onelayer(X_t, F) + U
    return dX_dt, U

def subgrid_component(X_curr, X_prev, dt, F):
    """Returns the subgrid-scale component of the one layer model"""
    return (X_curr - X_prev) / dt - dX_dt_onelayer(X_prev, F)  


# Define classes for L96 models
class L96Base:
    """Base class for Lorenz '96 models"""
    def __init__(self, dt=0.001, device='cpu', requires_grad=False):
        """
        Initialize base L96 model
        
        Args:
            dt (float): time step for integration
        """
        self.dt = dt
        self.time = 0.0
        self.X = None
        self.device = device
        self.requires_grad = requires_grad
    
    def _torch(self, X_0):
        """Convert input to torch tensor if needed"""
        if not isinstance(X_0, torch.Tensor):
            X_0 = torch.tensor(X_0, dtype=torch.float32, device=self.device, requires_grad=self.requires_grad)
        return X_0
    
    def get_solution(self):
        """Return current solution state"""
        return self.X, self.time

    def _initialize_history(self):
        """Initialize history arrays"""
        self.X_history = None
        self.time_history = None
    
    def _get_history(self):
        """Get history arrays"""
        return self.X_history, self.time_history

class L96OneLayer(L96Base):
    """Single layer Lorenz '96 model"""
    def __init__(self, X_0, dt=0.001, F=20, device='cpu', requires_grad=False):
        """
        Initialize one-layer model
        
        Args:
            X_0 (array-like): Initial conditions for X variables
            dt (float): time step for integration
            F (float): Forcing parameter
        """
        super().__init__(dt=dt, device=device, requires_grad=requires_grad)
        self.X = self._torch(X_0)
        self.K = len(X_0)
        self.F = self._torch(F)
        self._initialize_history()
    
    def iterate(self, T):
        """Iterate model forward in time"""
        nt = int(T/self.dt)
        X = torch.zeros((nt, self.K))
        time = torch.zeros(nt)

        X[0] = self.X
        time[0] = self.time
        for t in range(1,nt):
            X[t] = RK2_step(X[t-1], dX_dt_onelayer, self.dt, F=self.F)
            time[t] = time[t-1] + self.dt
        
        self.X = X[-1]
        self.time = time[-1]
        
        # Update history
        if self.X_history is None:
            self.X_history = X
            self.time_history = time
        else:
            self.X_history = torch.concat((self.X_history, X), axis=0)
            self.time_history = torch.concat((self.time_history, time), axis=0)
        
        return X, time


class L96TwoLayer(L96Base):
    """Two-layer Lorenz '96 model"""
    def __init__(self, X_0, Y_0, dt=0.001, F=20, c=10, b=10, h=1.0, device='cpu'):
        """
        Initialize two-layer model
        
        Args:
            X_0 (array-like): Initial conditions for X variables
            Y_0 (array-like): Initial conditions for Y variables
            dt (float): time step for integration
            F (float): Forcing parameter
            c (float): Coupling parameter
            b (float): Scale parameter
            h (float): Coupling coefficient
        """
        super().__init__(dt=dt, device=device)
        self.X = self._torch(X_0)
        self.Y = self._torch(Y_0)
        self.K = len(X_0)
        if len(Y_0) % self.K != 0:
            raise ValueError(f"Number of Y variables must be a multiple of K, \
             but K = {self.K} and len(Y_0) = {len(Y_0)}")
        self.J = len(Y_0) // self.K
        self.F = self._torch(F)
        self.c = self._torch(c)
        self.b = self._torch(b)
        self.h = self._torch(h)
        self._initialize_history()

    def _initialize_history(self):
        """Initialize history arrays"""
        self.X_history = None
        self.Y_history = None
        self.U_history = None
        self.time_history = None
        
    def _get_history(self):
        """Get history arrays"""
        return self.X_history, self.Y_history, self.U_history, self.time_history
    
    def iterate(self, T):
        """
        Iterates the two layer model
        """
        nt = int(T/self.dt)
        X = torch.zeros((nt, self.K))
        U = torch.zeros((nt, self.K))
        Y = torch.zeros((nt, self.K*self.J))

        time = torch.zeros(nt)

        Y[0] = self.Y
        X[0] = self.X
        time[0] = 0
        for t in range(1,nt):
            X[t], Y[t], U[t] = RK4_step_twolayer(X[t-1], Y[t-1], dX_dt_twolayer, dY_dt, self.dt,
                                                F=self.F, c=self.c, b=self.b, h=self.h, J=self.J, K=self.K)
            time[t] = time[t-1] + self.dt
        
        self.X = X[-1]
        self.Y = Y[-1]
        self.U = U[-1]
        self.time = time[-1]

        # Update history
        if self.X_history is None:
            self.X_history = X
            self.Y_history = Y
            self.U_history = U
            self.time_history = time
        else:
            self.X_history = torch.concatenate((self.X_history, X), axis=0)
            self.Y_history = torch.concatenate((self.Y_history, Y), axis=0)
            self.U_history = torch.concatenate((self.U_history, U), axis=0)
            self.time_history = torch.concatenate((self.time_history, time), axis=0)
        
        return X, Y, U, time
    
    
class L96OneLayerParam(L96OneLayer):
    """Parameterized single-layer Lorenz '96 model"""
    def __init__(self, X_0, param_func, dt=0.001, F=20, device='cpu', requires_grad=False):
        """
        Initialize parameterized one-layer model
        
        Args:
            X_0 (array-like): Initial conditions for X variables
            param_func (callable): Parameterization function
            dt (float): time step for integration
            F (float): Forcing parameter
        """
        super().__init__(X_0, dt=dt, F=F, device=device, requires_grad=requires_grad)
        self.param_func = param_func

    def iterate(self, T):
        """Iterate model forward in time"""
        nt = int(T/self.dt)
        X = torch.zeros((nt, self.K))
        U = torch.zeros((nt, self.K))
        time = torch.zeros(nt)

        X[0] = self.X
        time[0] = self.time
        for t in range(1, nt):
            U[t] = self.param_func(X[t-1])
            X[t] = RK2_step(X[t-1], dX_dt_onelayer, self.dt, F=self.F, U=U[t])
            time[t] = time[t-1] + self.dt
        
        self.X = X[-1]
        self.U = U[-1]
        self.time = time[-1]
        
        # Update history
        if self.X_history is None:
            self.X_history = X
            self.U_history = U
            self.time_history = time
        else:
            self.X_history = torch.concatenate((self.X_history, X), axis=0)
            self.U_history = torch.concatenate((self.U_history, U), axis=0)
            self.time_history = torch.concatenate((self.time_history, time), axis=0)
        
        return X, U, time
            
    def iterate_torch(self, T):
        """Iterate model forward in time"""
        nt = int(T/self.dt)
        X = self.X
        time = self.time
        for t in range(1,nt):
            U = self.param_func(X)
            X = RK2_step(X, dX_dt_onelayer, self.dt, F=self.F, U=U)
            time = time + self.dt
        
        self.X = X
        self.time = time
        
