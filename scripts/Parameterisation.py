import numpy as np
from collections import OrderedDict
import torch 

import pyro

class ParameterisationBase():
    """ Base class parameteristion"""
    def __init__(self, pyro_model, phi=0., res=0., N=10):
        self.pyro_model = pyro_model
        self.phi = phi
        self.res = res
        self.phi2 = phi**2
        self.sqrt_1_minus_phi2 = np.sqrt(1-self.phi2)
        self.N = N
        
    def param_sample(self):
        raise NotImplementedError

    def reset_param(self):
        """ Resets parameterisation by resetting residual from previous timestep to zero.
        Needed if using AR1 parameterisation and restarting with new initial conditions/ensembles. """
        self.res = 0

    def WN_param_epistemic(self, x):
        raise NotImplementedError

    def WN_param_aleatoric(self, x):
        raise NotImplementedError

    def WN_param_both(self, x):        
        raise NotImplementedError
        
    def AR1_param_epistemic(self, x):
        raise NotImplementedError

    def AR1_param_aleatoric(self, x):
        raise NotImplementedError

    def AR1_param_both(self, x):        
        raise NotImplementedError

class Parameterisation_VI():
    """General parameterisation class for Homoscedastic BNN learned via variational inference.
    Can do white noise or AR1 - epistemic, aleatoric or both."""
    def __init__(self, pyro_model, guide, sigma = 0., phi=0., res=0., N=10):
        """Initialise parameterisation with any parameters needed
        Args: 
        Sigma = stochastic std / sigma from guide
        phi = Lag1 autocorrelation. if 1, error is same for each timestep, no stochasicity, if 0, whitenoise.
        err = initial error to start with for AR1 process"""
        ParameterisationBase.__init__(self, pyro_model, phi=phi, res=res, N=N)
        self.guide = guide
        self.sigma = sigma

        # Draw random parameters for first call 
        self.guide_params = self.param_sample()
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.guide_params)

        # Calculate mean params
        self.param_samples_mean = self.guide.median()
        self.mean_NN = self.pyro_model.get_fixed_param_NN(self.param_samples_mean)
        self.mean_NN.eval()
        
        print(f"Set up AR1 Parameterisation from guide")

    def param_sample(self):
        """Returns new samples from guide"""
        return self.guide()
    
    def reset_param(self):
        """ Resets parameterisation by resetting residual from previous timestep to zero.
        Needed if using AR1 parameterisation and restarting with new initial conditions/ensembles. """
        self.res = 0

    def aleatoric_only(self, x):
        """Aleatoric only - keep parameters fixed at median """
        with torch.no_grad():
            det = self.mean_NN(x.unsqueeze(-1)).squeeze()
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * self.sigma * np.random.randn(x.shape[0])
        return det + self.res

    def keep_epistemic_fixed(self, x):
        """Can be used for Epistemic if sigma=0 or Both if sigma=learned sigma- 
        keep parameters fixed at their values"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * self.sigma * np.random.randn(x.shape[0])
        return det + self.res

    def estimate_epistemic_sigma(self, x, N=100):
        """Estimate the variance using Monte Carlo estimate to integrate over parameters (weights)"""
        if N==1:
            with torch.no_grad():
                det = self.mean_NN(x.unsqueeze(-1)).squeeze()
                f = self.fixed_param_NN(x.unsqueeze(-1)).squeeze() - self.mean_NN(x.unsqueeze(-1)).squeeze()
            sigma2 = (f - det)**2
            return sigma2

        f = torch.zeros((N, x.shape[0]))
        for n in range(N):
            self.sample_guide_params()
            with torch.no_grad():
                f[n] = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        sigma2 = torch.var(f, axis=0)
        return sigma2
    
    def AR1_param(self, x):
        """General AR1, can include aleatoric and epistemic with flags"""
        with torch.no_grad():
            det = self.mean_NN(x.unsqueeze(-1)).squeeze()
        sigma2 = torch.zeros(x.shape[0])
        if self.aleatoric:
            sigma2 = sigma2 + self.sigma**2
        if self.epistemic:
            if self.N==1:
                self.sample_guide_params()
                with torch.no_grad():
                    f = self.fixed_param_NN(x.unsqueeze(-1)).squeeze() 
                res = (f - det)
                self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * res
                return det + self.res
            sigma2 = sigma2 + self.estimate_epistemic_sigma(x, N=self.N)    
        sigma = torch.sqrt(sigma2)
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * sigma * np.random.randn(x.shape[0])
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
            f_det =  self.mean_NN(x.unsqueeze(-1)).squeeze()
        res = f_t - f_det
        # Update residual for next timestep
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * res
        return f_det + self.res



class Parameterisation_VI_Heteroscedastic(ParameterisationBase):
    """General parameterisation class for Heteroscedastic BNN learned via variational inference.
    Can do white noise or AR1 - epistemic, aleatoric or both."""
    def __init__(self, pyro_model, guide, phi=0., res=0., N=2):
        """Initialise parameterisation from variational inference.
        Args: 
        pyro_model = model
        guide = guide
        phi = Lag1 autocorrelation. if 1, error is same for each timestep, no stochasicity, if 0, whitenoise.
        err = initial error to start with for AR1 process"""
        ParameterisationBase.__init__(self, pyro_model, phi=phi, res=res, N=N)

        self.guide = guide

        # Draw random parameters for first call 
        self.guide_params = self.param_sample()
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.guide_params)

        # Calculate mean params
        self.param_samples_mean = self.guide.median()
        self.mean_NN = self.pyro_model.get_fixed_param_NN(self.param_samples_mean)
        self.mean_NN.eval()
        print(f"Set up AR1 Parameterisation from guide")
        
    def param_sample(self):
        """Returns new samples from guide"""
        return self.guide()
    
    def reset_param(self):
        """ Resets parameterisation by resetting residual from previous timestep to zero.
        Needed if using AR1 parameterisation and restarting with new initial conditions/ensembles. """
        self.res = 0
        
    def set_fixed_param_NN(self):
        """Sets fixed param NN, e.g., for fixed PPE simulations"""
        self.guide_params = self.param_sample()
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.guide_params)

    def WN_param_epistemic(self, x):
        param_samples = self.param_sample()
        NN = self.pyro_model.get_fixed_param_NN(param_samples)
        with torch.no_grad():
            out = NN(x.unsqueeze(-1))
        mu, _  = out.chunk(2, dim=-1)
        return mu.squeeze()

    def WN_param_aleatoric(self, x):
        with torch.no_grad():
            out = self.mean_NN(x.unsqueeze(-1)).squeeze()
        mu, sigma = out.chunk(2, dim=-1)
        sigma = torch.exp(sigma) + self.pyro_model.eps
        return mu.squeeze() + sigma.squeeze() * np.random.randn(x.shape[0])

    def WN_param_both(self, x):
        param_samples = self.param_sample()
        NN = self.pyro_model.get_fixed_param_NN(param_samples)
        with torch.no_grad():
            out = NN(x.unsqueeze(-1))
        mu, sigma  = out.chunk(2, dim=-1)
        sigma = torch.exp(sigma) + self.pyro_model.eps
        return mu.squeeze() + sigma.squeeze() * np.random.randn(x.shape[0])
    
    def AR1_param_aleatoric(self, x):
        """AR1 that samples aleatoric only - keep parameters fixed at median """
        with torch.no_grad():
            out = self.mean_NN(x.unsqueeze(-1)).squeeze()
        mu, sigma = out.chunk(2, dim=-1)
        sigma = torch.exp(sigma) + self.pyro_model.eps
        # y_t =  mu + residual = mu + phi*eps_{t-1} + sqrt(1-phi^2) * sigma * rand(0,1))
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * sigma.squeeze() * np.random.randn(x.shape[0])
        y_t = mu.squeeze() + self.res
        return y_t
    
    def AR1_param_epistemic(self, x):
        """AR1 that samples epistemic only, estimate variance from at least 2 samples """
        for n in range(self.N):
            param_samples = self.param_sample()
            NN = self.pyro_model.get_fixed_param_NN(param_samples)
            with torch.no_grad():
                out = NN(x.unsqueeze(-1))
            mean_n, _  = out.chunk(2, dim=-1)
            if n==0:
                means = mean_n
            else:
                means = torch.concat((means, mean_n), dim=1)
        # Mu is the mean of the mean
        mu = means.mean(dim=1)
        # epistemic is variance of conditional mean: Var_Θ (E[Y│X,Θ])
        sigma_epistemic = means.std(dim=1)
        # y_t =  mu + residual = mu + phi*eps_{t-1} + sqrt(1-phi^2) * sigma * rand(0,1))
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * sigma_epistemic * np.random.randn(x.shape[0])
        y_t = mu + self.res
        return y_t

     def AR1_param_both(self, x):
        """AR1 that samples both aleatoric and epistemic by drawing one sample """
        param_samples = self.param_sample()
        NN = self.pyro_model.get_fixed_param_NN(param_samples)
        with torch.no_grad():
            out = NN(x.unsqueeze(-1))
        mu, sigma  = out.chunk(2, dim=-1)
        sigma = (torch.exp(sigma) + self.pyro_model.eps)**2
        sigma = sigma.squeeze()
        mu = mu.squeeze()
        # y_t =  mu + residual = mu + phi*eps_{t-1} + sqrt(1-phi^2) * sigma * rand(0,1))
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * np.sqrt(sigma) * np.random.randn(x.shape[0])
        y_t = mu + self.res
        return y_t

    def fixed_param_epistemic(self, x):
        """Run with parameters fixed, sample epistemic only"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        mean, sigma = det.chunk(2, dim=-1)
        return mean
        
    def fixed_param_both(self, x):
        """Run with parameters fixed, sample both (Use AR1 for aleatoric)"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        mean, sigma = det.chunk(2, dim=-1)
        sigma = torch.exp(sigma) + self.pyro_model.eps
        self.res = self.phi * self.res +  self.sqrt_1_minus_phi2 * sigma.squeeze() * np.random.randn(x.shape[0])
        return mean.squeeze() + self.res



class Parameterisation_MCMC_Heteroscedastic():
    """General parameterisation class for Heteroscedastic BNN learned via MCMC.
    Can do white noise or AR1 - epistemic, aleatoric or both."""
    def __init__(self, pyro_model, posterior_samples, phi=0., res=0., N=2):
        """Initialise parameterisation from variational inference.
        Args: 
        pyro_model = model
        guide = guide
        phi = Lag1 autocorrelation. if 1, error is same for each timestep, no stochasicity, if 0, whitenoise.
        err = initial error to start with for AR1 process"""
        ParameterisationBase.__init__(self, pyro_model, phi=phi, res=res, N=N)

        self.posterior_samples = posterior_samples
        # Extract num_samples
        self.num_samples = list(posterior_samples.values())[0].shape[0]

        # Draw random parameters for first call 
        self.mcmc_params = self.param_sample()
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.mcmc_params)

        # Calculate mean params
        self.param_samples_mean = OrderedDict( (k, v.mean(dim=0)) for k, v in self.posterior_samples.items())
        self.mean_NN = self.pyro_model.get_fixed_param_NN(self.param_samples_mean)
        self.mean_NN.eval()
                
        print(f"Set up AR1 Parameterisation, with {self.num_samples} posterior samples stored")
    
    
    def reset_param(self):
        """ Resets parameterisation by resetting residual from previous timestep to zero.
        Needed if using AR1 parameterisation and restarting with new initial conditions/ensembles. """
        self.res = 0

    def param_sample(self, r=None):
        """Returns new sample dictionary of parameters from posterior samples"""
        if r == None:
            r = np.random.randint(0, self.num_samples)
        return {k: v[r] for k, v in self.posterior_samples.items()}

    def set_fixed_param_NN(self):
        """Sets fixed param NN, e.g., for fixed PPE simulations"""
        self.mcmc_params = self.param_sample()
        self.fixed_param_NN = self.pyro_model.get_fixed_param_NN(self.mcmc_params)

    def WN_param_epistemic(self, x):
        param_samples = self.param_sample()
        NN = self.pyro_model.get_fixed_param_NN(param_samples)
        with torch.no_grad():
            out = NN(x.unsqueeze(-1))
        mu, _  = out.chunk(2, dim=-1)
        return mu.squeeze()

    def WN_param_aleatoric(self, x):
        with torch.no_grad():
            out = self.mean_NN(x.unsqueeze(-1))
        mu, sigma = out.chunk(2, dim=-1)
        sigma = torch.exp(sigma) + self.pyro_model.eps
        return mu.squeeze() + sigma.squeeze()*np.random.randn(x.shape[0])

    def WN_param_both(self, x):
        param_samples = self.param_sample()
        NN = self.pyro_model.get_fixed_param_NN(param_samples)
        with torch.no_grad():
            out = NN(x.unsqueeze(-1))
        mu, sigma  = out.chunk(2, dim=-1)
        sigma = (torch.exp(sigma) + self.pyro_model.eps)
        return mu.squeeze() + sigma.squeeze()*np.random.randn(x.shape[0])

    def AR1_param_aleatoric(self, x):
        """AR1 that samples aleatoric only - keep parameters fixed at median """
        with torch.no_grad():
            out = self.mean_NN(x.unsqueeze(-1)).squeeze()
        mu, sigma = out.chunk(2, dim=-1)
        sigma = torch.exp(sigma) + self.pyro_model.eps
        # y_t =  mu + residual = mu + phi*eps_{t-1} + sqrt(1-phi^2) * sigma * rand(0,1))
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * sigma.squeeze() * np.random.randn(x.shape[0])
        y_t = mu.squeeze() + self.res
        return y_t
    
    def AR1_param_epistemic(self, x):
        """AR1 that samples epistemic only, estimate variance from at least 2 samples """
        for n in range(self.N):
            param_samples = self.param_sample()
            NN = self.pyro_model.get_fixed_param_NN(param_samples)
            with torch.no_grad():
                out = NN(x.unsqueeze(-1))
            mean_n, _  = out.chunk(2, dim=-1)
            if n==0:
                means = mean_n
            else:
                means = torch.concat((means, mean_n), dim=1)
        # Mu is the mean of the mean
        mu = means.mean(dim=1)
        # epistemic is variance of conditional mean: Var_Θ (E[Y│X,Θ])
        sigma_epistemic = means.std(dim=1)
        # y_t =  mu + residual = mu + phi*eps_{t-1} + sqrt(1-phi^2) * sigma * rand(0,1))
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * sigma_epistemic * np.random.randn(x.shape[0])
        y_t = mu + self.res
        return y_t

     def AR1_param_both(self, x):
        """AR1 that samples both aleatoric and epistemic by drawing one sample """
        param_samples = self.param_sample()
        NN = self.pyro_model.get_fixed_param_NN(param_samples)
        with torch.no_grad():
            out = NN(x.unsqueeze(-1))
        mu, sigma  = out.chunk(2, dim=-1)
        sigma = (torch.exp(sigma) + self.pyro_model.eps)**2
        sigma = sigma.squeeze()
        mu = mu.squeeze()
        # y_t =  mu + residual = mu + phi*eps_{t-1} + sqrt(1-phi^2) * sigma * rand(0,1))
        self.res = self.phi * self.res + self.sqrt_1_minus_phi2 * np.sqrt(sigma) * np.random.randn(x.shape[0])
        y_t = mu + self.res
        return y_t
    
    def fixed_param_epistemic(self, x):
        """Run with parameters fixed, sample epistemic only"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        mean, sigma = det.chunk(2, dim=-1)
        return mean
        
    def fixed_param_both(self, x):
        """Run with parameters fixed, sample both (Use AR1 for aleatoric)"""
        with torch.no_grad():
            det = self.fixed_param_NN(x.unsqueeze(-1)).squeeze()
        mean, sigma = det.chunk(2, dim=-1)
        sigma = torch.exp(sigma) + self.pyro_model.eps
        self.res = self.phi * self.res +  self.sqrt_1_minus_phi2 * sigma.squeeze() * np.random.randn(x.shape[0])
        return mean.squeeze() + self.res
