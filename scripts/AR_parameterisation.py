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
        self.phi2 = phi**2
        self.sqrt_1_minus_phi2 = np.sqrt(1-self.phi2)

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
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * self.sigma * np.random.randn(x.shape[0])
        return det + self.res

    def keep_epistemic_fixed(self, x):
        """Can be used for Epistemic if sigma=0 or Both if sigma=learned sigma- 
        keep parameters fixed at their values"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * self.sigma * np.random.randn(x.shape[0])
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
        with torch.no_grad():
            f_det =  self.deterministic_NN(x.unsqueeze(-1)).squeeze()
        res = f_t - f_det
        # Update residual for next timestep
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * res
        return f_det + self.res



class ParameterisationAR1_Heteroscedastic(ParameterisationAR1):
    def __init__(self, pyro_model, guide, include_sigma = False, phi=0., res=0.):
        """Initialise parameterisation with any parameters needed
        Args: 
        Sigma = stochastic std / sigma from guide
        phi = Lag1 autocorrelation. if 1, error is same for each timestep, no stochasicity, if 0, whitenoise.
        err = initial error to start with for AR1 process"""
        ParameterisationAR1.__init__(self, pyro_model, guide, sigma = 0., phi = phi, res = res)
        self.include_sigma = include_sigma
        
    def sample_guide_params(self):
        self.guide_params = self.guide() 
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.guide_params)

    def aleatoric_only(self, x):
        """Aleatoric only - keep parameters fixed at median """
        with torch.no_grad():
            det = self.deterministic_NN(x.unsqueeze(-1)).squeeze()
        mean, sigma = det.chunk(2, dim=-1)
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * sigma.squeeze() * np.random.randn(x.shape[0])
        return mean.squeeze() + self.res

    def keep_epistemic_fixed(self, x):
        """Can be used for Epistemic if sigma=0 or Both if sigma=learned sigma- 
        keep parameters fixed at their values"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        mean, sigma = det.chunk(2, dim=-1)
        self.res = self.phi * self.res + (self.include_sigma) * (self.sqrt_1_minus_phi2 * sigma.squeeze() * np.random.randn(x.shape[0]))
        return mean.squeeze() + self.res

    def epistemic_AR1(self, x):
        """Epistemic uncertainty treated as AR1, can do epistemic only if sigma=0 or both if sigma is set"""
        # Sample new parameters for time t
        self.sample_guide_params()
        with torch.no_grad():
            # Epistemic part
            f_t = self.fixed_param_NN(x.unsqueeze(-1)).squeeze() 
            mean_t, sigma_t = f_t.chunk(2, dim=1)
        f_t = mean_t.squeeze() + self.include_sigma * sigma_t.squeeze() * np.random.randn(x.shape[0])

        with torch.no_grad():
            f_det =  self.deterministic_NN(x.unsqueeze(-1)).squeeze()
            mean_det, sigma_det = f_det.chunk(2, dim=1)
        res = f_t - mean_det.squeeze()

        # Update residual for next timestep
        self.res = self.phi * self.res + np.sqrt(1-self.phi**2) * res  
        return mean_det.squeeze() + self.res