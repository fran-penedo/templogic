from scipy import optimize
from llt import set_llt_pars, SimpleModel, split_groups
from stl import robustness
import numpy as np
import math


def optimize_inf_gain(traces, primitive, rho):
    # [t0, t1, t3, pi]
    maxt = max(np.amax(traces.get_sindex(-1), 1))
    lower = [0, 0, 0, min(np.amin(traces.get_sindex(primitive.index), 1))]
    upper = [maxt, maxt, maxt,
             max(np.amax(traces.get_sindex(primitive.index), 1))]
    models = [SimpleModel(signal) for signal in traces.signals]
    args = (primitive, models, rho, traces, maxt)

    res = optimize.differential_evolution(inf_gain, bounds=zip(lower, upper),
                                          args=args)
    return primitive, res.fun

def inf_gain(theta, *args):
    primitive = args[0]
    models = args[1]
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = args[2]
    traces = args[3]
    maxt = args[4]

    if theta[1] < theta[0] or theta[1] + theta[2] > maxt:
        return np.inf

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    lrho = [[robustness(primitive, model) for model in models]]
    if prev_rho is not None:
        lrho.append(prev_rho)
    rho_labels = zip(np.amin(lrho, 0), traces.labels)
    sat, unsat = split_groups(rho_labels, lambda x: x[0]>= 0)

    # compute IG
    stotal = sum(np.abs(zip(*rho_labels)[0]))
    ig = entropy(rho_labels) - inweights(sat, stotal) * entropy(sat) - \
        inweights(unsat, stotal) * entropy(unsat)

    return ig

def inweights(part, stotal):
    if len(part) == 0:
        return 0
    return sum(np.abs(zip(*part)[0])) / stotal

def entropy(part):
    if len(part) == 0:
        return 0

    spart = float(sum(np.abs(part[0])))
    w_p = sum([p[0] for p in part if p[1] >= 0]) / spart
    w_n = - sum([p[0] for p in part if p[1] < 0]) / spart
    if w_p <= 0 or w_n <= 0:
        return 0
    else:
        return - w_p * math.log(w_p) - w_n * math.log(w_n)

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



