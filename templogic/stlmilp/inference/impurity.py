""" Impurity measure definition and optimization module.

Currently defines information gain.

Author: Francisco Penedo (franp@bu.edu)

"""
from __future__ import division, absolute_import, print_function

import math
import logging
from bisect import bisect_right

import numpy as np  # type: ignore
import scipy.optimize as opt  # type: ignore

from .llt import SimpleModel
from ..stl import robustness
from templogic.util import split_groups

logger = logging.getLogger(__name__)


def optimize_impurity(
    traces,
    primitive,
    rho,
    disp=False,
    optimizer_args=None,
    times=None,
    interpolate=False,
    tinter=None,
    impurity=None,
):
    """ Optimizes the impurity function given for the given labeled traces.

    """
    optimizer_args_def = {
        "popsize": 10,
        "maxiter": 10,
        "mutation": 0.7,
        "init": "latinhypercube",
        "workers": 1,
        "polish": False,
    }
    if optimizer_args is not None:
        optimizer_args_def.update(optimizer_args)
    if impurity is None:
        impurity = ext_inf_gain
    # [t0, t1, t3, pi]
    maxt = max(np.amax(traces.get_sindex(-1), 1))
    # Might not be needed since this is for forces and the last one is not
    # relevant
    # if times is not None:
    #     maxt = maxt + times[1] # Add one more to reach the ends
    lower, upper = primitive.parameter_bounds(
        maxt,
        min(np.amin(traces.get_sindex(primitive.index), 1)),
        max(np.amax(traces.get_sindex(primitive.index), 1)),
    )
    models = [SimpleModel(signal, interpolate, tinter) for signal in traces.signals]
    args = DEArgs(primitive, models, rho, traces, maxt, times)

    # Optimize over t0, v1, v3, pi, where v1 / maxt = t1 - t0 / maxt - t0 and
    # v3 / maxt = t3 / maxt - t1
    if len(traces) < 50:
        optimizer_args_def["workers"] = 1
    res = opt.differential_evolution(
        impurity,
        bounds=list(zip(lower, upper)),
        args=(args,),
        disp=disp,
        **optimizer_args_def
    )
    theta = _transform_pars(res.x, maxt, times)
    primitive.set_llt_pars(theta)
    return primitive, res.fun


class DEArgs(object):
    def __init__(self, primitive, models, rho, traces, maxt, times=None):
        self.primitive = primitive
        self.models = models
        self.rho = rho
        self.traces = traces
        self.maxt = maxt
        self.times = times


def _transform_pars(theta, maxt, times):
    # t0, v1, v3, pi -> t0, t1, t3, pi
    if len(theta) == 4:
        t0, t1, t3, pi = theta
        if maxt > 0:
            t1 = t0 + (maxt - t0) * t1 / maxt
            t3 = (maxt - t1) * t3 / maxt
            t0, t1, t3 = [_round_t(t, times) for t in [t0, t1, t3]]
        else:
            t0 = t1 = t3 = 0.0

        return [t0, t1, t3, pi]
    elif len(theta) == 3:
        t0, t1, pi = theta
        if maxt > 0:
            t1 = t0 + (maxt - t0) * t1 / maxt
            t0, t1 = [_round_t(t, times) for t in [t0, t1]]
        else:
            t0 = t1 = 0.0

        return [t0, t1, pi]
    else:
        raise ValueError()


def _round_t(t, times):
    if times is None:
        return t
    else:
        i = bisect_right(times, t) - 1
        return times[i]


def inf_gain(theta, *args):
    """ Obtains the negative of information gain of the sample theta.

    The extra fixed arguments are defined as:
        args = [primitive, models, prev_rho, traces, maxt]
    where primitive is the formula to optimize, models is a list of SimpleModel
    objects associated with each trace for the signal index defined in the
    primitive, prev_rho is the robustness of each trace up until the current
    node, traces is a Traces object and maxt is the maximum sampled time.
    """
    args = args[0]
    primitive = args.primitive
    models = args.models
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = args.rho
    traces = args.traces
    maxt = args.maxt
    times = args.times

    theta = _transform_pars(theta, maxt, times)

    # if theta[1] < theta[0] or theta[1] + theta[2] > maxt:
    #     print 'bad'
    #     return np.inf

    primitive.set_llt_pars(theta)

    rho = [robustness(primitive, model) for model in models]
    rho = [0.0 if np.isclose(0.0, r, atol=1e-5) else r for r in rho]
    if np.any(np.isclose(0.0, rho, atol=1e-5)):
        penalty = 100.0
    else:
        penalty = 0.0
    lrho = [rho]

    if prev_rho is not None:
        lrho.append(prev_rho)
    rho_labels = list(zip(np.amin(lrho, 0), traces.labels))
    sat, unsat = split_groups(rho_labels, lambda x: x[0] >= 0)

    # # compute IG
    # # Sum of absolute value of the robustness for all traces
    # stotal = sum(np.abs(zip(*rho_labels)[0]))
    # # FIXME should probably take into account the domain of the signals
    # if np.isclose(0.0, stotal, atol=1e-5):
    #     ig = -np.nan
    # else:
    stotal = len(rho_labels)
    ig = (
        _entropy(rho_labels)
        - (len(sat) / stotal) * _entropy(sat)
        - (len(unsat) / stotal) * _entropy(unsat)
    )

    return -ig


def _entropy(part):
    if len(part) == 0:
        return 0.0

    w_p = len([p for p in part if p[1] >= 0]) / float(len(part))
    w_n = len([p for p in part if p[1] < 0]) / float(len(part))

    if w_p <= 0 or w_n <= 0:
        return 0.0
    else:
        return -w_p * math.log(w_p) - w_n * math.log(w_n)


def ext_inf_gain(theta, *args):
    """ Obtains the negative of extended information gain of the sample theta.

    The extra fixed arguments are defined as:
        args = [primitive, models, prev_rho, traces, maxt]
    where primitive is the formula to optimize, models is a list of SimpleModel
    objects associated with each trace for the signal index defined in the
    primitive, prev_rho is the robustness of each trace up until the current
    node, traces is a Traces object and maxt is the maximum sampled time.
    """
    args = args[0]
    primitive = args.primitive
    models = args.models
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = args.rho
    traces = args.traces
    maxt = args.maxt
    times = args.times

    theta = _transform_pars(theta, maxt, times)

    # if theta[1] < theta[0] or theta[1] + theta[2] > maxt:
    #     print 'bad'
    #     return np.inf

    primitive.set_llt_pars(theta)

    rho = [robustness(primitive, model) for model in models]
    rho = [0.0 if np.isclose(0.0, r, atol=1e-5) else r for r in rho]
    if np.any(np.isclose(0.0, rho, atol=1e-5)):
        penalty = 100.0
    else:
        penalty = 0.0
    lrho = [rho]

    if prev_rho is not None:
        lrho.append(prev_rho)
    rho_labels = list(zip(np.amin(lrho, 0), traces.labels))
    sat, unsat = split_groups(rho_labels, lambda x: x[0] >= 0)

    # compute IG
    # Sum of absolute value of the robustness for all traces
    stotal = sum(np.abs(list(zip(*rho_labels))[0]))
    # FIXME should probably take into account the domain of the signals
    if np.isclose(0.0, stotal, atol=1e-5):
        ig = 0.0
    else:
        ig = (
            _ext_entropy(rho_labels)
            - _ext_inweights(sat, stotal) * _ext_entropy(sat)
            - _ext_inweights(unsat, stotal) * _ext_entropy(unsat)
        )

    return -ig + penalty


def _ext_inweights(part, stotal):
    if len(part) == 0:
        return 0
    return sum(np.abs(list(zip(*part))[0])) / stotal


def _ext_entropy(part):
    if len(part) == 0:
        return 0

    spart = float(sum(np.abs(list(zip(*part))[0])))
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
        return -w_p * math.log(w_p) - w_n * math.log(w_n)


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
    t3 = (1 - t1) * t3

    return [t0, t1, t3, pi]
