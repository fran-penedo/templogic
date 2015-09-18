from scipy import optimize
from llt import set_llt_pars, SimpleModel
import numpy as np


def optimize_inf_gain(signals, primitive, rho):
    # [t0, t1, t3, pi]
    maxt = max(np.amax(signals[:,-1], 1))
    lower = [0, 0, 0, min(np.amin(signals[:, primitive.index], 1))]
    upper = [maxt, maxt, maxt, max(np.amax(signals[:, primitive.index], 1))]
    models = [SimpleModel(signal) for signal in signals]
    args = (primitive, models, rho)

    res = optimize.differential_evolution(inf_gain, bounds=zip(lower, upper),
                                          args=args)
    return primitive, res.fun

def inf_gain(theta, *args):
    primitive = args[0]
    models = args[1]
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = args[2]

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    rho = np.amin([prev_rho[:,primitive.index], [robustness(primitive, model)
                              for model in models]], 1)

    # compute IG
    ig = 0

    return ig



def optimize_inf_gain_skel(signals, primitive, robustness):
    # [t0, t1, t3, pi]
    maxt = max(np.amax(signals[:,-1], 1))
    lower = [0, 0, 0, min(np.amin(signals[:, primitive.index], 1))]
    upper = [maxt, maxt, maxt, max(np.amax(signals[:, primitive.index], 1))]
    args = (primitive, signals, robustness)

    res = optimize.differential_evolution(inf_gain_skel, bounds=zip(lower, upper),
                                          args=args)
    return primitive, res.fun

def inf_gain_skel(theta, *args):
    primitive = args[0]
    signals = args[1]
    # May be None, TODO check. Can't do it up in the stack
    robustness = args[2]

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    return - theta[0] - theta[1] - theta[2] - theta[3] - primitive.index



