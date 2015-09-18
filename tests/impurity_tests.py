from impurity import *
from llt import make_llt_primitives
import numpy as np


def opt_inf_gain_skel_test():
    signals = np.array([
        [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
        [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
    ])
    primitive = make_llt_primitives(signals)[0]
    robustness = None

    np.testing.assert_almost_equal(
        optimize_inf_gain_skel(signals, primitive, robustness)[1],
        -16, 2)
