import pytest
import numpy as np
import matplotlib.pyplot as plt
from L96_model import iterate_onelayer, iterate_twolayer

@pytest.fixture
def model_params():
    """Fixture for common model parameters"""
    return {
        'F': 20,
        'c': 10,
        'b': 10,
        'h': 1.0,
        'J': 32,
        'K': 8,
        # Add other parameters as needed
    }

def check_slope(time, data, tol=0.1, varname=''):
    """Check if the data is stationary over time"""
    m, c = np.polyfit(time, data, 1)
    assert np.abs(m) < tol, f"{varname} not stationary, slope={m}"


def test_stationary_distributions_onelayer(model_params, plot=False):
    """Test if the solution remains stable and stationary over long time scales"""
    X_0 = np.random.rand(model_params['K'])
    dt = 0.01
    T = 50
    spinup = 2
    # Test One Layer Model
    X, time = iterate_onelayer(X_0, dt, T, model_params['F'])

    # First test stability
    assert np.all(np.isfinite(X)), "One layer model unstable: X is not finite"

    # Distributions over longer time scales should remain roughly constant
    # Discard spinup
    X = X[int(spinup/dt):]
    time = time[int(spinup/dt):]

    # Calculate mean and std of X
    mean_X = np.mean(X, axis=1)
    std_X = np.std(X, axis=1)
    # These should be roughly constant, i.e. if we fit a line to the data, 
    # the slope should be close to zero
    check_slope(time, mean_X, tol=0.2, varname='mean_X')
    check_slope(time, std_X, tol=0.1, varname='std_X')

    if plot:
        plt.plot(time, mean_X, label='mean_X')
        plt.plot(time, std_X, label='std_X')
        plt.title('One Layer Model Distributions')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

def test_stationary_distributions_twolayer(model_params, plot=False):
    """Test if the solution remains stable and stationary over long time scales"""
    X_0 = np.random.rand(model_params['K'])
    Y_0 = np.random.rand(model_params['K']*model_params['J'])
    dt = 0.001
    T = 50
    spinup = 2
    # Test Two Layer Model
    X, Y, time, U = iterate_twolayer(X_0, Y_0, dt, T, 
    model_params['F'], 
    model_params['c'], 
    model_params['b'], 
    model_params['h'], 
    model_params['J'], 
    model_params['K'])

    # First test stability
    assert np.all(np.isfinite(X)), "Two layer model unstable: X is not finite"
    assert np.all(np.isfinite(Y)), "Two layer model unstable: Y is not finite"

    # Distributions over longer time scales should remain roughly constant
    # Discard spinup
    X = X[int(spinup/dt):]
    Y = Y[int(spinup/dt):]
    time = time[int(spinup/dt):]

    # Calculate mean and std of X and Y
    mean_X = np.mean(X, axis=1)
    std_X = np.std(X, axis=1)
    mean_Y = np.mean(Y, axis=1)
    std_Y = np.std(Y, axis=1)
    # These should be roughly constant, i.e. if we fit a line to the data, 
    # the slope should be close to zero
    print(time.shape, mean_X.shape, mean_Y.shape)
    
    check_slope(time, mean_X, tol=0.2, varname='mean_X')
    check_slope(time, std_X, tol=0.1, varname='std_X')
    check_slope(time, mean_Y, tol=0.2, varname='mean_Y')
    check_slope(time, std_Y, tol=0.1, varname='std_Y')

    if plot:
        plt.plot(time, mean_X, label='mean_X')
        plt.plot(time, mean_Y, label='mean_Y')
        plt.plot(time, std_X, label='std_X')
        plt.plot(time, std_Y, label='std_Y')
        plt.title('Two Layer Model Distributions')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

# Run with some different values for X_0, X_1, dt, T
@pytest.mark.parametrize("X_0, X_1, dt, T", [
    (np.array([-0.47217838,  6.15099598, 12.07293103, 10.02733354,  5.55605426,
       -7.74351186, -7.20291528,  1.53862845]),
     np.array([-0.43546637,  6.15911804, 12.13811173, 10.02898425,  5.38976478,
       -7.80342309, -7.15078015,  1.50799177]), 
       0.0001,
       0.001),
    (np.array([-7.54512685, -0.05364396, 14.61747508, 11.17113759,  0.28896955,
        7.09419   , 14.41910287, -2.79011643]),
    np.array([-7.48293987, -0.15385246, 14.62066656, 11.18383373,  0.23109528,
        7.106567  , 14.40414081, -2.95914369]),
        0.0001,
        0.001),
])
def test_known_solution_onelayer(model_params, X_0, X_1, dt, T, plot=True):
    """Test if the known solution is obtained for a given set of parameters. 
    Known solution generated from Euler integration with dt=0.0001, i.e. run for 10 time steps"""

    X, time = iterate_onelayer(X_0, dt, T, model_params['F'])
    
    if plot:
        plt.plot(X_0, label='X_0')
        plt.plot(X_1, label='X true')
        plt.plot(X[-1], label='X computed')
        plt.title('One Layer Model Known Solution')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
    
    # Compare last value of X to known solution, X_1
    assert np.allclose(X[-1], X_1, atol=1e-4), "One layer model solution not close to known solution!"

def test_zero_param_equals_onelayer(model_params):
    """Test if the zero parameterization equals the one layer model"""
    X_0 = np.random.rand(model_params['K'])
    dt = 0.001
    T = 10
    X, time = iterate_onelayer(X_0, dt, T, model_params['F'])
    X_param, time_param = iterate_onelayer_param(X_0, dt, T, lambda x: 0)
    assert np.allclose(X, X_param, atol=1e-4), "Zero parameterization does not equal one layer model!"
    

@pytest.mark.parametrize("dt", [0.1, 0.01, 0.001])
def test_convergence(dt, model_params):
    """Test convergence with different time steps"""
    X_0 = np.random.rand(model_params['K'])
    Y_0 = np.random.rand(model_params['K']*model_params['J'])
    dt = 0.001
    T = 10
    spinup = 1
    # Test One Layer Model
    X, time = iterate_onelayer(X_0, dt, T, model_params['F'])
    