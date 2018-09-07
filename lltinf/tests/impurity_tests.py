from __future__ import division, absolute_import, print_function

import logging
import unittest

import numpy as np

from lltinf import inference, impurity, llt

class TestSTL(unittest.TestCase):

    def test_opt_inf_gain_skel(self):
        traces = inference.Traces([
            [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
            [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
        ], [1,1])
        primitive = llt.make_llt_primitives(traces.signals)[0]
        robustness = None

        np.testing.assert_almost_equal(
            impurity.optimize_inf_gain_skel(traces, primitive, robustness)[1],
            -16, 2)

    def test_inf_gain(self):
        traces = inference.Traces([
            [[1,2,3,4], [1,2,3,3.5], [1,2,3,4]],
            [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
            [[1,2,3,4], [1,2,3,5], [1,2,3,4]],
            [[1,2,3,4], [1,2,3,5.5], [1,2,3,4]]
        ], [1, 1, -1, -1])
        models = [llt.SimpleModel(signal) for signal in traces.signals]
        primitive_good = llt.LLTFormula(True, 1, llt.LE)
        primitive_bad = llt.LLTFormula(True, 0, llt.LE)
        rho = None
        maxt = 4

        print(impurity.inf_gain([1, 4, 0, 4.5], primitive_good, models, rho, traces, maxt))
        print(impurity.inf_gain([1, 4, 0, 4.5], primitive_bad, models, rho, traces, maxt))

    def test_inf_gain2(self):
        traces = inference.Traces(
            [[[-3.7972648],
            [-1.22366531],
            [ 0.        ]],

            [[ 1.4999998 ],
            [-1.22366532],
            [ 0.        ]],

            [[-0.49999975],
            [-1.22366533],
            [ 0.        ]]]
            , [1, -1, -1])
        models = [llt.SimpleModel(signal) for signal in traces.signals]
        primitive_good = llt.LLTFormula(True, 0, llt.LE)
        primitive_bad = llt.LLTFormula(True, 1, llt.LE)
        rho = np.array([1.0000000e+00, 3.3345704e-09, 1.0000000e+00])
        maxt = 0

        gain_good = impurity.inf_gain(
            [0, 0, 0, -1.0], primitive_good, models, rho, traces, maxt)
        gain_bad = impurity.inf_gain(
            [0, 0, 0, -1.22366], primitive_bad, models, rho, traces, maxt)

        self.assertLess(gain_bad, gain_good)

    def test_opt_inf_gain(self):
        traces = inference.Traces(
            [[[-3.7972648],
            [-1.22366531],
            [ 0.        ]],

            [[ 1.4999998 ],
            [-1.22366532],
            [ 0.        ]],

            [[-0.49999975],
            [-1.22366533],
            [ 0.        ]]]
            , [1, -1, -1])
        rho = np.array([1.0000000e+00, 3.3345704e-09, 1.0000000e+00])
        prims = llt.make_llt_primitives(traces.signals)
        # primitive_bad = llt.LLTFormula(False, 1, llt.LE)

        opts = [impurity.optimize_inf_gain(traces, primitive.copy(), rho) for
                primitive in prims]

        for p, imp in opts:
            print("{} ({})".format(p, imp))

        self.assertAlmostEqual(min(zip(*opts)[1]), 0.6)




