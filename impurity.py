from scipy import optimize
from llt import set_llt_pars, SimpleModel, split_groups
import numpy as np


def optimize_inf_gain(traces, primitive, rho):
    # [t0, t1, t3, pi]
    maxt = max(np.amax(traces.get_sindex(-1), 1))
    lower = [0, 0, 0, min(np.amin(traces.get_sindex(primitive.index), 1))]
    upper = [maxt, maxt, maxt,
             max(np.amax(traces.get_sindex(primitive.index), 1))]
    models = [SimpleModel(signal) for signal in traces.signals]
    args = (primitive, models, rho, traces)

    res = optimize.differential_evolution(inf_gain, bounds=zip(lower, upper),
                                          args=args)
    return primitive, res.fun

def inf_gain(theta, *args):
    primitive = args[0]
    models = args[1]
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = args[2]
    traces = args[3]

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    rho = np.amin([prev_rho[:,primitive.index], [robustness(primitive, model)
                              for model in models]], 1)
    sat, unsat = split_groups(zip(rho, traces.labels), lambda x: x[0]>= 0)

    # compute IG
    stotal = sum(abs(rho[0]))
    ig = entropy(rho) - inweights(sat, stotal) * entropy(sat) - \
        inweights(unsat, stotal) * entropy(unsat)

    return ig

def inweights(part, stotal):
    return sum(abs(part)) / stotal

def entropy(part):
    spart = sum(abs(part[0]))
    w_p = sum([p for p in part[1] if p >= 0]) / spart
    w_n = - sum([p for p in part[1] if p < 0]) / spart
    return - w_p * math.log(w_p)






def optimize_inf_gain_skel(traces, primitive, rho):
    # [t0, t1, t3, pi]
    maxt = max(np.amax(traces.get_sindex(-1), 1))
    lower = [0, 0, 0, min(np.amin(traces.get_sindex(primitive.index), 1))]
    upper = [maxt, maxt, maxt,
             max(np.amax(traces.get_sindex(primitive.index), 1))]
    models = [SimpleModel(signal) for signal in traces.signals]
    args = (primitive, models, rho)

    res = optimize.differential_evolution(inf_gain_skel, bounds=zip(lower, upper),
                                          args=args)
    return primitive, res.fun

def inf_gain_skel(theta, *args):
    primitive = args[0]
    traces = args[1]
    # May be None, TODO check. Can't do it up in the stack
    robustness = args[2]

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    return - theta[0] - theta[1] - theta[2] - theta[3] - primitive.index



