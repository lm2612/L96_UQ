import numpy as np

import torch 

import pyro

class ParameterisationAR1():
    def __init__(self, pyro_model, guide, sigma = 0., phi=0., res=0.):
        """Initialise parameterisation with any parameters needed
        Args: 
        Sigma = stochastic std / sigma from guide
        phi = Lag1 autocorrelation. if 1, error is same for each timestep, no stochasicity, if 0, whitenoise.
        err = initial error to start with for AR1 process"""
        #self.fixed_param_NN = fixed_param_NN
        self.pyro_model = pyro_model
        self.guide = guide
        self.phi = phi
        self.res = res
        self.sigma = sigma

        # draw parameters for first call
        self.guide_params = self.guide() 
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.guide_params)

        self.deterministic_NN = self.pyro_model.get_fixed_param_NN(self.guide.median())
        self.deterministic_NN.eval()

    def sample_guide_params(self):
        self.guide_params = self.guide() 
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.guide_params)

    def aleatoric_only(self, x):
        """Aleatoric only - keep parameters fixed at median """
        with torch.no_grad():
            det = self.deterministic_NN(x.unsqueeze(-1)).squeeze()
        self.res = self.phi * self.res + np.sqrt(1-self.phi**2) * self.sigma * np.random.randn(x.shape[0])
        return det + self.res

    def keep_epistemic_fixed(self, x):
        """Can be used for Epistemic if sigma=0 or Both if sigma=learned sigma- 
        keep parameters fixed at their values"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        self.res = self.phi * self.res + np.sqrt(1-self.phi**2) * self.sigma * np.random.randn(x.shape[0])
        return det + self.res

    def epistemic_AR1(self, x):
        """Epistemic uncertainty treated as AR1, can do epistemic only if sigma=0 or both if sigma is set"""
        # Sample new parameters for time t
        self.sample_guide_params()
        with torch.no_grad():
            # Update aleatoric part (will be zero if sigma is zero)
            err = self.sigma * np.random.randn(x.shape[0])
            # Epistemic part
            f_t = self.fixed_param_NN(x.unsqueeze(-1)).squeeze() + err
        #out = f_t + self.phi * self.res 
        with torch.no_grad():
            f_det =  self.deterministic_NN(x.unsqueeze(-1)).squeeze()
        res = f_t - f_det
        #out = f_det + self.phi * self.res + np.sqrt(1-self.phi**2) * res
        # Update residual for next timestep
        self.res = self.phi * self.res + np.sqrt(1-self.phi**2) * res
        return f_det + self.res