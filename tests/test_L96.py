import pytest
import numpy as np
import matplotlib.pyplot as plt
from L96_model import iterate_onelayer, iterate_twolayer, iterate_onelayer_param
from L96_model import L96OneLayer, L96TwoLayer, L96OneLayerParam
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
def test_known_solution_onelayer(model_params, X_0, X_1, dt, T, plot=False):
    """Test if the known solution is obtained for a given set of parameters. 
    Known solution generated from Euler integration with dt=0.0001, i.e. run for 10 time steps"""

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
    assert np.allclose(X[-1], X_1, atol=1e-4), "One layer model solution not close to known solution!"

@pytest.mark.parametrize("X_0, Y_0, X_1, Y_1, dt, T", [
    (np.array([0.61403837, 0.59142689, 0.3162574 , 0.10823514, 0.36171908,
        0.09530081, 0.38544378, 0.76615813]),
     np.array([0.97918559, 0.55664291, 0.34915038, 0.65405296, 0.58215156,
        0.25316062, 0.07886977, 0.29522418, 0.3038067 , 0.55527964,
        0.78827774, 0.20081169, 0.29010625, 0.97812732, 0.88896415,
        0.88733431, 0.40648443, 0.74341888, 0.25620629, 0.38674371,
        0.10413267, 0.44082973, 0.34704276, 0.44758962, 0.33394645,
        0.05285271, 0.39152664, 0.96567635, 0.89395632, 0.34916724,
        0.02969935, 0.83429772, 0.27716024, 0.80099109, 0.56295282,
        0.36823627, 0.34601919, 0.25873183, 0.80704512, 0.00821438,
        0.49378867, 0.01309798, 0.16100236, 0.55336577, 0.09465636,
        0.79938718, 0.12374229, 0.67819483, 0.64927265, 0.04273152,
        0.07760373, 0.17126757, 0.03020249, 0.02474618, 0.3946603 ,
        0.62787152, 0.64639184, 0.1217859 , 0.23777758, 0.99056588,
        0.03363804, 0.10432705, 0.06413768, 0.63632884, 0.45301396,
        0.67360147, 0.45838019, 0.47559935, 0.28638102, 0.56484991,
        0.8423154 , 0.80035264, 0.91264846, 0.81190569, 0.0559694 ,
        0.60264991, 0.34538722, 0.74694638, 0.63496383, 0.44715942,
        0.01527629, 0.0589732 , 0.28606855, 0.8488301 , 0.94403342,
        0.73153541, 0.99966885, 0.11404939, 0.13195117, 0.91137175,
        0.23019858, 0.07646171, 0.6484334 , 0.54321884, 0.73605334,
        0.00950698, 0.5723647 , 0.63253756, 0.07051748, 0.25928851,
        0.298577  , 0.46318886, 0.63526711, 0.11448957, 0.0303837 ,
        0.28532634, 0.18607136, 0.75982428, 0.78608098, 0.86230343,
        0.83117673, 0.96334652, 0.82116406, 0.10034166, 0.45457665,
        0.78774471, 0.51684012, 0.04238455, 0.09462038, 0.83169794,
        0.11136508, 0.10379736, 0.99760292, 0.82009977, 0.62683203,
        0.35364636, 0.10662574, 0.56704074, 0.79800455, 0.3203717 ,
        0.84484709, 0.44056366, 0.47926918, 0.01669803, 0.35018978,
        0.80387842, 0.57678621, 0.58666999, 0.89426775, 0.68254574,
        0.69931721, 0.91390704, 0.24825544, 0.5404637 , 0.47374565,
        0.97284319, 0.96167037, 0.46138143, 0.55786395, 0.68696993,
        0.23237788, 0.68501154, 0.91860462, 0.18259584, 0.30586084,
        0.92108277, 0.07181092, 0.22122568, 0.17512288, 0.05726226,
        0.10236682, 0.99895661, 0.86646311, 0.75448034, 0.22417206,
        0.64805495, 0.37907368, 0.02628882, 0.56921611, 0.14242394,
        0.78778137, 0.15478728, 0.29745428, 0.02596273, 0.16498101,
        0.27834143, 0.79451857, 0.30334393, 0.58037068, 0.30013944,
        0.59901992, 0.00436934, 0.65370449, 0.14961417, 0.7243096 ,
        0.5049029 , 0.57134371, 0.69538596, 0.42855958, 0.25962556,
        0.55841442, 0.38978276, 0.3472892 , 0.11087682, 0.96289871,
        0.59946625, 0.20149525, 0.91604983, 0.60770307, 0.77391165,
        0.91264294, 0.97148825, 0.03347214, 0.55957258, 0.98001345,
        0.30178068, 0.94157517, 0.93175811, 0.89079763, 0.73864251,
        0.95838055, 0.74625602, 0.52728939, 0.68518302, 0.7948892 ,
        0.62781458, 0.79012823, 0.7645127 , 0.72598222, 0.78846718,
        0.90704543, 0.12755626, 0.75707276, 0.43997828, 0.1795916 ,
        0.28060464, 0.71841337, 0.56582015, 0.44168712, 0.50060176,
        0.78078195, 0.02173887, 0.23523878, 0.78614798, 0.11258002,
        0.02830507, 0.93546732, 0.91122577, 0.86054409, 0.63857966,
        0.60247513, 0.90413619, 0.82839581, 0.25817327, 0.75067438,
        0.28639582, 0.4917779 , 0.74981519, 0.55801852, 0.77462234,
        0.49045438, 0.33178966, 0.90480679, 0.52830687, 0.31136211,
        0.17506787])
     np.array([0.6248002 , 0.60373657, 0.32653084, 0.11948477, 0.37189654,
        0.10720121, 0.39376319, 0.77624452]), 
     np.array([ 0.9708319 ,  0.55937566,  0.34714861,  0.65389072,  0.58624816,
         0.25332783,  0.07822437,  0.28798636,  0.29069668,  0.5565145 ,
         0.7875387 ,  0.1980509 ,  0.25965887,  0.95018186,  0.90671233,
         0.88653522,  0.42530249,  0.74077258,  0.26611443,  0.38451553,
         0.10468233,  0.43394261,  0.34784262,  0.45026654,  0.33291141,
         0.04254728,  0.35422166,  0.96275883,  0.90455007,  0.34800657,
         0.03229578,  0.82149387,  0.28540563,  0.7955568 ,  0.56811213,
         0.37141541,  0.33974587,  0.26938981,  0.80361928,  0.02551925,
         0.49180949,  0.01289488,  0.1584344 ,  0.54839066,  0.10932601,
         0.79298704,  0.12784744,  0.67781938,  0.64783706,  0.04442671,
         0.07762564,  0.17085596,  0.03014057,  0.01476486,  0.37595228,
         0.63291107,  0.64588022,  0.11793301,  0.24065318,  0.98670748,
         0.03795737,  0.10257766,  0.05445596,  0.62164149,  0.45613096,
         0.67039948,  0.46460713,  0.47249803,  0.27650372,  0.54362921,
         0.82593989,  0.79797703,  0.93495809,  0.80937207,  0.06804142,
         0.58987806,  0.34261257,  0.74095801,  0.64655636,  0.44574196,
         0.01576689,  0.0490973 ,  0.25248444,  0.82573121,  0.93481133,
         0.76455539,  0.99851522,  0.11419744,  0.12689703,  0.90810466,
         0.23018613,  0.06728227,  0.62977589,  0.56120436,  0.73296724,
         0.01217325,  0.56821671,  0.63081168,  0.07404067,  0.25314969,
         0.28963442,  0.46621445,  0.63471515,  0.11450139,  0.02951205,
         0.2785271 ,  0.16892219,  0.73329056,  0.77960919,  0.85169032,
         0.82916891,  0.98537048,  0.81993319,  0.10054528,  0.4382654 ,
         0.79350791,  0.51590272,  0.04096918,  0.0917482 ,  0.82812849,
         0.11030898,  0.07251655,  0.97375407,  0.8340434 ,  0.63533049,
         0.35245162,  0.09534667,  0.5568283 ,  0.79069916,  0.33191542,
         0.83832427,  0.45610453,  0.47740171,  0.01199774,  0.32923545,
         0.79442111,  0.57209396,  0.5801245 ,  0.88688845,  0.67904369,
         0.71352591,  0.91198035,  0.25777204,  0.52365572,  0.45354919,
         0.96880294,  0.96595996,  0.46634178,  0.56261485,  0.68287939,
         0.22454579,  0.68404793,  0.91775642,  0.18194133,  0.30899997,
         0.91749941,  0.07884864,  0.22056601,  0.17484517,  0.05410416,
         0.06661212,  0.9690628 ,  0.88768554,  0.75350208,  0.23388284,
         0.64856939,  0.37756479,  0.03190898,  0.56185958,  0.15583132,
         0.78335085,  0.16401772,  0.29619648,  0.02604305,  0.15508388,
         0.27205943,  0.78701989,  0.31430829,  0.57400199,  0.3137042 ,
         0.59629514,  0.01708839,  0.64619976,  0.15359755,  0.7119314 ,
         0.50329291,  0.57119815,  0.69822968,  0.42831869,  0.25942039,
         0.55454437,  0.39493061,  0.3433086 ,  0.09994016,  0.9563726 ,
         0.59747625,  0.20047145,  0.89723569,  0.60464325,  0.75600287,
         0.93988825,  0.96795378,  0.0333497 ,  0.54583216,  0.97093728,
         0.30236212,  0.91368766,  0.9352076 ,  0.88622292,  0.74150208,
         0.96136382,  0.74953806,  0.52382873,  0.6788462 ,  0.78870915,
         0.62624228,  0.78366425,  0.76101851,  0.71801294,  0.8084763 ,
         0.90349917,  0.14251047,  0.75309178,  0.44196218,  0.17584534,
         0.26755537,  0.71154446,  0.5676992 ,  0.43538398,  0.51296465,
         0.77796579,  0.02192295,  0.23143613,  0.78412782,  0.1123395 ,
        -0.00369752,  0.89777361,  0.91821249,  0.86592864,  0.63494707,
         0.59260545,  0.91254915,  0.82696947,  0.27510068,  0.74496284,
         0.28553871,  0.48100469,  0.73979947,  0.56461467,  0.77649685,
         0.48679792,  0.3289771 ,  0.90168054,  0.53625327,  0.30703678,
         0.16404338]),
       0.0001,
       0.001),
])
def test_known_solution_twolayer(model_params, X_0, Y_0, X_1, Y_1, dt, T, plot=True):
    """Test if the known solution is obtained for a given set of parameters. 
    Known solution generated from Euler integration with dt=0.0001, i.e. run for 10 time steps"""
    model = L96TwoLayer(X_0, Y_0, 
                        dt=dt,
                        F=model_params['F'], 
                        c=model_params['c'], 
                        b=model_params['b'], 
                        h=model_params['h'])
    X, Y, U, time = model.iterate(T)

    if plot:
        plt.plot(X_0, label='X_0')
        plt.plot(X_1, label='X true')
        plt.plot(X[-1], label='X computed')
        plt.title('Two Layer Model Known Solution')
        plt.xlabel('Time')
        plt.legend()
        plt.show()
    
    # Compare last value of X to known solution, X_1
    assert np.allclose(X[-1], X_1, atol=1e-4), "Two layer model solution for X not close to known solution!"
    assert np.allclose(Y[-1], Y_1, atol=1e-4), "Two layer model solution for Y not close to known solution!"

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
    