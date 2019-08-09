from __future__ import division, absolute_import, print_function

import logging
import unittest
import pprint

import numpy as np
import numpy.testing as npt

from lltinf import llt

class TestLLT(unittest.TestCase):

    def test_primitives(self):
        signals = [
            [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
            [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
        ]
        prims = llt.make_llt_primitives(signals)
        prims[0].set_llt_pars([1, 2, 3, 4])
        pprint.pprint(prims)
        pprint.pprint(prims[0].copy())

    def test_split_groups(self):
        x = [1,-1,2,-2]
        p, n = llt.split_groups(x, lambda t: t >= 0)
        np.testing.assert_array_equal(p, [1, 2])
        np.testing.assert_array_equal(n, [-1, -2])

