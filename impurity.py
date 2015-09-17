from scipy import optimize
from llt import set_llt_pars
import numpy as np

def optimize_inf_gain(signals, primitive, robustness):
    # [t0, t1, t3, pi]
    maxt = max(signals[-1])
    theta0 = [0, 0, 0, 0]
    lower = [0, 0, 0, min(signals[primitive.index])]
    upper = [maxt, maxt, maxt, min(signals[primitive.index])]
    args = [primitive, signals, robustness]

    res = optimize.anneal(inf_gain, theta0, args=args, lower=lower, upper=upper)
    return primitive, res[1]

def inf_gain(theta, *args):
    primitive = args[0]
    signals = args[1]
    # May be None, TODO check. Can't do it up in the stack
    robustness = args[2]

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    # TODO use actual function
    return - theta[0] - theta[1] - theta[2] - theta[3] - primitive.index


def optimize_inf_gain_skel(signals, primitive, robustness):
    # [t0, t1, t3, pi]
    maxt = max(np.amax(signals[:,-1], 1))
    lower = [0, 0, 0, min(np.amin(signals[:, primitive.index], 1))]
    upper = [maxt, maxt, maxt, max(np.amax(signals[:, primitive.index], 1))]
    args = (primitive, signals, robustness)

    res = optimize.differential_evolution(inf_gain_skel, bounds=zip(lower, upper),
                                          args=args)
    return primitive, res.x

def inf_gain_skel(theta, *args):
    primitive = args[0]
    signals = args[1]
    # May be None, TODO check. Can't do it up in the stack
    robustness = args[2]

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    return - theta[0] - theta[1] - theta[2] - theta[3] - primitive.index



