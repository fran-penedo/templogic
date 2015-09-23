from impurity import *
from lltinf import Traces
from llt import make_llt_primitives, LLTFormula
from stl import LE
import numpy as np


def opt_inf_gain_skel_test():
    traces = Traces([
        [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
        [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
    ], [1,1])
    primitive = make_llt_primitives(traces.signals)[0]
    robustness = None

    np.testing.assert_almost_equal(
        optimize_inf_gain_skel(traces, primitive, robustness)[1],
        -16, 2)

def opt_inf_gain_test():
    traces = Traces([
        [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
        [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
    ], [1,-1])
    models = [SimpleModel(signal) for signal in traces.signals]
    primitive = LLTFormula(True, 1, LE)
    rho = None
    maxt = 4

    print inf_gain([1, 4, 0, 4.5], primitive, models, rho, traces, maxt)

