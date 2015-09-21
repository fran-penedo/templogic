from llt import *
import pprint
import numpy as np

def primitives_test():
    signals = [
        [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
        [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
    ]
    prims = make_llt_primitives(signals)
    set_llt_pars(prims[0], 1, 2, 3, 4)
    pprint.pprint(prims)

    pprint.pprint(prims[0].copy())

def split_groups_test():
    x = [1,-1,2,-2]
    p, n = split_groups(x, lambda t: t >= 0)
    np.testing.assert_array_equal(p, [1, 2])
    np.testing.assert_array_equal(n, [-1, -2])

