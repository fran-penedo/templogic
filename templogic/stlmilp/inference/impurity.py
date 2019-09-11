""" Impurity measure definition and optimization module.

Currently defines information gain.

Author: Francisco Penedo (franp@bu.edu)

"""
from __future__ import division, absolute_import, print_function

import math
import logging
from typing import Tuple, Sequence, Iterable, Any, Generic, TypeVar
from abc import ABC, abstractmethod

import numpy as np
import scipy.optimize as opt  # type: ignore

from .llt import Primitive
from ..stl import STLModel
from templogic.util import split_groups

logger = logging.getLogger(__name__)

U = TypeVar("U")


class ImpurityDataSet(ABC, Generic[U]):
    @property
    @abstractmethod
    def time_bounds(self) -> Tuple[float, float]:
        pass

    @property
    @abstractmethod
    def data_bounds(self) -> Sequence[Tuple[float, float]]:
        pass

    def models(self, interpolate: bool, tinter: float) -> Iterable[STLModel]:
        return [self.model(signal, interpolate, tinter) for signal in self.signals]

    @abstractmethod
    def model(self, signal: Any, interpolate: bool, tinter: float) -> STLModel:
        pass

    @property
    @abstractmethod
    def labels(self) -> Sequence[int]:
        pass

    @property
    @abstractmethod
    def signals(self) -> Sequence[U]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


def optimize_impurity(
    traces: ImpurityDataSet,
    primitive: Primitive,
    rho,
    disp=False,
    optimizer_args=None,
    times=None,
    interpolate: bool = False,
    tinter=None,
    impurity=None,
) -> Tuple[Primitive, float]:
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
    # maxt = max(np.amax(traces.get_sindex(-1), 1))
    # mint = min(np.amin(traces.get_sindex(-1), 1))
    # Might not be needed since this is for forces and the last one is not
    # relevant
    # if times is not None:
    #     maxt = maxt + times[1] # Add one more to reach the ends
    # lower, upper = primitive.parameter_bounds(
    #     (mint, maxt),
    #     min(np.amin(traces.get_sindex(primitive.index), 1)),
    #     max(np.amax(traces.get_sindex(primitive.index), 1)),
    # )
    bounds = primitive.parameter_bounds(traces.time_bounds, traces.data_bounds)
    models = list(traces.models(interpolate, tinter))
    args = DEArgs(primitive, models, rho, traces, times)

    # Optimize over t0, v1, v3, pi, where v1 / maxt = t1 - t0 / maxt - t0 and
    # v3 / maxt = t3 / maxt - t1
    if len(traces) < 50:
        optimizer_args_def["workers"] = 1
    res = opt.differential_evolution(
        impurity, bounds=bounds, args=(args,), disp=disp, **optimizer_args_def
    )
    primitive.set_pars(res.x, traces.time_bounds, times)
    return primitive, res.fun


class DEArgs(object):
    def __init__(
        self, primitive: Primitive, models, rho, traces: ImpurityDataSet, times=None
    ):
        self.primitive = primitive
        self.models = models
        self.rho = rho
        self.traces = traces
        self.times = times


def inf_gain(theta: Tuple, *args: DEArgs) -> float:
    """ Obtains the negative of information gain of the sample theta.

    The extra fixed arguments are defined as:
        args = [primitive, models, prev_rho, traces, maxt]
    where primitive is the formula to optimize, models is a list of SimpleModel
    objects associated with each trace for the signal index defined in the
    primitive, prev_rho is the robustness of each trace up until the current
    node, traces is a Traces object and maxt is the maximum sampled time.
    """
    l_args = args[0]
    primitive = l_args.primitive
    models = l_args.models
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = l_args.rho
    traces = l_args.traces
    times = l_args.times

    primitive.set_pars(theta, traces.time_bounds, times)

    rho = [primitive.score(model) for model in models]
    rho = [0.0 if np.isclose(0.0, r, atol=1e-5) else r for r in rho]
    # if np.any(np.isclose(0.0, rho, atol=1e-5)):
    #     penalty = 100.0
    # else:
    #     penalty = 0.0
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


def _entropy(part) -> float:
    if len(part) == 0:
        return 0.0

    w_p = len([p for p in part if p[1] >= 0]) / float(len(part))
    w_n = len([p for p in part if p[1] < 0]) / float(len(part))

    if w_p <= 0 or w_n <= 0:
        return 0.0
    else:
        return -w_p * math.log(w_p) - w_n * math.log(w_n)


def ext_inf_gain(theta: Tuple, *args: DEArgs):
    """ Obtains the negative of extended information gain of the sample theta.

    The extra fixed arguments are defined as:
        args = [primitive, models, prev_rho, traces, maxt]
    where primitive is the formula to optimize, models is a list of SimpleModel
    objects associated with each trace for the signal index defined in the
    primitive, prev_rho is the robustness of each trace up until the current
    node, traces is a Traces object and maxt is the maximum sampled time.
    """
    l_args = args[0]
    primitive = l_args.primitive
    models = l_args.models
    # May be None, TODO check. Can't do it up in the stack
    prev_rho = l_args.rho
    traces = l_args.traces
    times = l_args.times

    # if theta[1] < theta[0] or theta[1] + theta[2] > maxt:
    #     print 'bad'
    #     return np.inf

    primitive.set_pars(theta, traces.time_bounds, times)

    rho = [primitive.score(model) for model in models]
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


def _ext_inweights(part, stotal) -> float:
    if len(part) == 0:
        return 0
    return sum(np.abs(list(zip(*part))[0])) / stotal


def _ext_entropy(part) -> float:
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
