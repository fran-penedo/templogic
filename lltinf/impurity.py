"""
Impurity measure definition and optimization module.

Currently defines information gain.

Author: Francisco Penedo (franp@bu.edu)

"""
import math
import logging

import numpy as np
from scipy import optimize

from lltinf.llt import set_llt_pars, SimpleModel, split_groups
from stlmilp.stl import robustness

logger = logging.getLogger(__name__)

def optimize_inf_gain(traces, primitive, rho, disp=False):
    """
    Optimizes the extended information gain for the given labeled traces.

    """
    # [t0, t1, t3, pi]
    maxt = max(np.amax(traces.get_sindex(-1), 1))
    lower = [0, 0, 0, min(np.amin(traces.get_sindex(primitive.index), 1))]
    upper = [maxt, maxt, maxt,
             max(np.amax(traces.get_sindex(primitive.index), 1))]
    models = [SimpleModel(signal) for signal in traces.signals]
    args = (primitive, models, rho, traces, maxt)

    # Optimize over t0, v1, v3, pi, where v1 / maxt = t1 - t0 / maxt - t0 and
    # v3 / maxt = t3 / maxt - t1
    res = optimize.differential_evolution(
        inf_gain, bounds=zip(lower, upper),
        args=args, popsize=10, maxiter=10,
        mutation=0.7, disp=disp,
        init='latinhypercube')
    return primitive, res.fun


def _transform_pars(theta, maxt):
    # Transform all arguments to be in [0, 1]
    t0, t1, t3, pi = theta
    if maxt > 0:
        t1 = t0 + (maxt - t0) * t1 / maxt
        t3 = (maxt - t1) * t3 / maxt
    else:
        t0 = t1 = t3 = 0.0

    return [t0, t1, t3, pi]

def inf_gain(theta, *args):
    """
    Function to optimize. Obtains the information gain of the sample theta.

    The extra fixed arguments are defined as:
        args = [primitive, models, prev_rho, traces, maxt]
    where primitive is the formula to optimize, models is a list of SimpleModel
    objects associated with each trace for the signal index defined in the
    primitive, prev_rho is the robustness of each trace up until the current
    node, traces is a Traces object and maxt is the maximum sampled time.
    """
    primitive = args[0]
    models = args[1]
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = args[2]
    traces = args[3]
    maxt = args[4]

    theta = _transform_pars(theta, maxt)

    if theta[1] < theta[0] or theta[1] + theta[2] > maxt:
        print 'bad'
        return np.inf

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    rho = [robustness(primitive, model) for model in models]
    rho = [0.0 if np.isclose(0.0, r, atol=1e-5) else r for r in rho]
    if np.any(np.isclose(0.0, rho, atol=1e-1)):
        penalty = 100.0
    else:
        penalty = 0.0
    lrho = [rho]

    if prev_rho is not None:
        lrho.append(prev_rho)
    rho_labels = zip(np.amin(lrho, 0), traces.labels)
    sat, unsat = split_groups(rho_labels, lambda x: x[0]>= 0)

    # compute IG
    # Sum of absolute value of the robustness for all traces
    stotal = sum(np.abs(zip(*rho_labels)[0]))
    # FIXME should probably take into account the domain of the signals
    if np.isclose(0.0, stotal, atol=1e-5):
        ig = -np.nan
    else:
        ig = _entropy(rho_labels) - _inweights(sat, stotal) * _entropy(sat) - \
            _inweights(unsat, stotal) * _entropy(unsat)

    return -ig + penalty

def _inweights(part, stotal):
    if len(part) == 0:
        return 0
    return sum(np.abs(zip(*part)[0])) / stotal

def _entropy(part):
    if len(part) == 0:
        return 0

    spart = float(sum(np.abs(zip(*part)[0])))
    # Revert to counting when all rho = 0
    if spart == 0:
        w_p = len([p for p in part if p[1] >= 0]) / float(len(part))
        w_n = len([p for p in part if p[1] < 0]) / float(len(part))
    else:
        w_p = sum([abs(p[0]) for p in part if p[1] >= 0]) / spart
        w_n = sum([abs(p[0]) for p in part if p[1] < 0]) / spart

    if w_p <= 0 or w_n <= 0:
        return 0
    else:
        return - w_p * math.log(w_p) - w_n * math.log(w_n)

# Dummy functions to test the optimization structure

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


# Unused

def constrained_sample(theta_scaled):
    # all in [0, 1]
    t0, t1, t3, pi = theta_scaled
    if t0 > t1:
        t0, t1 = t1, t0
    if t1 + t3 > 1:
        t3 = 1 - t1

    return [t0, t1, t3, pi]

def constrained_sample_init(theta_scaled):
    # all in [0, 1]
    t0, t1, t3, pi = theta_scaled
    t1 = t0 + (1 - t0) * t1
    t3 = (1-t1) * t3

    return [t0, t1, t3, pi]
