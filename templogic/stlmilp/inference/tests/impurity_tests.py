from __future__ import division, absolute_import, print_function

import logging
import unittest

import numpy as np  # type: ignore
import numpy.testing as npt  # type: ignore

from .. import inference, impurity, llt


class TestSTL(unittest.TestCase):
    def test_ext_inf_gain(self) -> None:
        traces = inference.Traces(
            [
                [[1, 2, 3, 4], [1, 2, 3, 3.0], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 3.5], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 5.0], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 6.0], [1, 2, 3, 4]],
            ],
            [1, 1, -1, -1],
        )
        models = [llt.SimpleModel(signal) for signal in traces.signals]
        primitive_good = llt.LLTFormula(True, 1, "<")
        primitive_bad = llt.LLTFormula(True, 0, "<")
        rho = None
        maxt = 4

        npt.assert_almost_equal(
            impurity.ext_inf_gain(
                [1, 4, 0, 4.5], impurity.DEArgs(primitive_good, models, rho, traces)
            ),
            -0.6869615765973234,
        )

        npt.assert_almost_equal(
            impurity.ext_inf_gain(
                [1, 4, 0, 4.5], impurity.DEArgs(primitive_bad, models, rho, traces)
            ),
            0.0,
        )

    def test_inf_gain(self):
        traces = inference.Traces(
            [
                [[1, 2, 3, 4], [1, 2, 3, 3.0], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 3.5], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 5.0], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 6.0], [1, 2, 3, 4]],
            ],
            [1, 1, -1, -1],
        )
        models = [llt.SimpleModel(signal) for signal in traces.signals]
        primitive_good = llt.LLTFormula(True, 1, "<")
        primitive_bad = llt.LLTFormula(True, 0, "<")
        rho = None
        maxt = 4

        npt.assert_almost_equal(
            impurity.inf_gain(
                [1, 4, 0, 4.5], impurity.DEArgs(primitive_good, models, rho, traces)
            ),
            -0.69314718056,
        )

        npt.assert_almost_equal(
            impurity.inf_gain(
                [1, 4, 0, 4.5], impurity.DEArgs(primitive_bad, models, rho, traces)
            ),
            0.0,
        )

    def test_ext_inf_gain2(self):
        traces = inference.Traces(
            [
                [[-3.7972648], [-1.22366531], [0.0]],
                [[1.4999998], [-1.22366532], [0.0]],
                [[-0.49999975], [-1.22366533], [0.0]],
            ],
            [1, -1, -1],
        )
        models = [llt.SimpleModel(signal) for signal in traces.signals]
        primitive_good = llt.LLTFormula(True, 0, "<")
        primitive_bad = llt.LLTFormula(True, 1, "<")
        rho = np.array([1.0000000e00, 3.3345704e-09, 1.0000000e00])
        maxt = 0

        gain_good = impurity.ext_inf_gain(
            [0, 0, 0, -1.0], impurity.DEArgs(primitive_good, models, rho, traces)
        )
        gain_bad = impurity.ext_inf_gain(
            [0, 0, 0, -1.22366], impurity.DEArgs(primitive_bad, models, rho, traces)
        )

        self.assertGreater(gain_bad, gain_good)

    def test_opt_inf_gain(self):
        traces = inference.Traces(
            [
                [[-3.7972648], [-1.22366531], [0.0]],
                [[1.4999998], [-1.22366532], [0.0]],
                [[-0.49999975], [-1.22366533], [0.0]],
            ],
            [1, -1, -1],
        )
        rho = np.array([1.0000000e00, 3.3345704e-09, 1.0000000e00])
        prims = llt.make_llt_primitives(traces.signals)
        # primitive_bad = llt.LLTFormula(False, 1, llt.LE)

        opts = [
            impurity.optimize_impurity(traces, primitive.copy(), rho)
            for primitive in prims
        ]

        for p, imp in opts:
            print("{} ({})".format(p, imp))

        self.assertAlmostEqual(min(list(zip(*opts))[1]), -0.6364488070531602, places=3)
