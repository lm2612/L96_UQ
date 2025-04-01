import pytest
import numpy as np
import matplotlib.pyplot as plt
from L96_model import L96OneLayer, L96TwoLayer, L96OneLayerParam, subgrid_component
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
    model = L96OneLayer(X_0, dt=dt, F=model_params['F'])
    X, time = model.iterate(T)

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
    model = L96TwoLayer(X_0, Y_0, 
                        dt=dt,
                        F=model_params['F'], 
                        c=model_params['c'], 
                        b=model_params['b'], 
                        h=model_params['h'])
    X, Y, U, time = model.iterate(T)

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
    # Test 1: K=4, True solution from M2Lines RK2
    (np.array([ 0.70120951, -3.95498642,  3.75292588,  8.49408818]),
    np.array([-2.49205159, -0.91396802,  2.81247061, 10.0322123 ]),
    0.001,
    0.1,
 ),
    # Test 2: K=8, True solution from M2Lines RK2
    (np.array([-6.45609487, 10.93808542,  4.67189148,  6.36765283,  4.52882792,
        -6.27244321,  4.02496443, 14.06263477]),
        np.array([10.45760188, 11.19159657,  7.99144926, -0.57251116,  1.72765392,
        -4.21062005, -0.59557727, 15.10761347]),
        0.001,
        0.1
    ),
    # Test 3: K=8, True solution from Euler integration (different integration method so diverges faster)
    (np.array([-7.54512685, -0.05364396, 14.61747508, 11.17113759,  0.28896955,
        7.09419   , 14.41910287, -2.79011643]),
    np.array([-7.48293987, -0.15385246, 14.62066656, 11.18383373,  0.23109528,
        7.106567  , 14.40414081, -2.95914369]),
        0.0001,
        0.001),
])
def test_known_solution_onelayer(model_params, X_0, X_1, dt, T, plot=False):
    """Test if the known solution is obtained for a given set of parameters. """
    model = L96OneLayer(X_0, 
                        dt=dt,
                        F=model_params['F'])
    X, time = model.iterate(T)  

    if plot:
        plt.plot(X_0, label='X_0')
        plt.plot(X_1, label='X true')
        plt.plot(X[-1], label='X computed')
        plt.title('One Layer Model Known Solution')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
    
    # Compare last value of X to known solution, X_1
    print(X[-1], X_1)
    assert np.allclose(X[-1], X_1, atol=1e-3), "One layer model solution not close to known solution!"

@pytest.mark.parametrize("X_0, Y_0, X_1, Y_1, dt, T", [
    # Test 1: K=4, J=2, True solution from M2Lines RK4
    (np.array([ 0.70120951, -3.95498642,  3.75292588,  8.49408818]),
     np.array([ 0.42710408,  0.29823534,  0.04808923, -0.1973897 , -0.21931402,
        -0.16434865,  0.17949164, -0.41263649,  0.16388677,  0.11594328,
        -0.18697502,  0.25416426,  0.45878876,  0.11530994,  0.08184334,
         0.56240793]),
    np.array([-2.44881807, -0.90278846,  2.834384  ,  9.90898454]),
    np.array([-0.39321021, -0.17469805, -0.15657025, -0.05767148, -0.15842552,
        -0.01098017, -0.03800284,  0.07806573, -0.3265347 ,  0.31957621,
         0.09033512, -0.14529296,  0.2868481 , -0.05385897,  0.89976186,
         0.10388236]),
    0.001, 
    0.1),
])
def test_known_solution_twolayer(model_params, X_0, Y_0, X_1, Y_1, dt, T, plot=False):
    """Test if the known solution is obtained for a given set of parameters. """
    K = X_0.shape[0]
    J = Y_0.shape[0] // K
    model = L96TwoLayer(X_0, Y_0, 
                        dt=dt,
                        F=model_params['F'], 
                        c=model_params['c'], 
                        b=model_params['b'], 
                        h=model_params['h'])
    X, Y, U, time = model.iterate(T)

    if plot:
        plt.plot(np.linspace(0, K-1, K), X_0, label='X_0')
        plt.plot(np.linspace(0, K-1, K), X_1, label='X true')
        plt.plot(np.linspace(0, K-1, K), X[-1], label='X computed')
        plt.plot(np.linspace(0, K-1, K*J), Y_0, label='Y_0')
        plt.plot(np.linspace(0, K-1, K*J), Y_1, label='Y true')
        plt.plot(np.linspace(0, K-1, K*J), Y[-1], label='Y computed')
        plt.title('Two Layer Model Known Solution')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
    
    # Compare last value of X to known solution, X_1
    print(Y_1, Y[-1])
    assert np.allclose(X[-1], X_1, atol=1e-5), "Two layer model solution for X not close to known solution!"
    assert np.allclose(Y[-1], Y_1, atol=1e-5), "Two layer model solution for Y not close to known solution!"

def test_zero_param_equals_onelayer(model_params):
    """Test if the zero parameterization equals the one layer model"""
    X_0 = np.random.rand(model_params['K'])
    dt = 0.001
    T = 10
    model = L96OneLayer(X_0, 
                        dt=dt,
                        F=model_params['F'])
    X, time = model.iterate(T)
    model_param = L96OneLayerParam(X_0, lambda x: 0, 
                                   dt=dt,
                                   F=model_params['F'])
    X_param, U, time_param = model_param.iterate(T)
    assert np.allclose(X, X_param, atol=1e-4), "Zero parameterization does not equal one layer model!"
    
def test_subgrid_component(model_params):
    """Test if the subgrid component is close to the true subgrid component (U)"""
    X_0 = np.random.rand(model_params['K'])
    Y_0 = np.random.rand(model_params['K']*model_params['J'])
    dt = 0.001
    F = 20
    T = 20
    spinup = 2
    # Run two layer model
    model = L96TwoLayer(X_0, Y_0, 
                        dt=dt,
                        F=F, 
                        c=model_params['c'], 
                        b=model_params['b'], 
                        h=model_params['h'])
    X, Y, U_true, time = model.iterate(T)
    # Discard spinup
    X = X[int(spinup/dt):]
    U_true = U_true[int(spinup/dt):]
    # Estimate subgrid component
    U_est = -subgrid_component(X[1:], X[:-1], dt, F)
    print(((U_est - U_true[1:])).max())
    # Only needs to be approximately close because we do not use Y to calculate U - use 10% tolerance
    assert np.allclose(U_est, U_true[1:], atol=2), "Subgrid component is not close to the true subgrid component!"
