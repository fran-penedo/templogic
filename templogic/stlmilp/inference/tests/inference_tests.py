from __future__ import division, absolute_import, print_function

import logging
import unittest
import pickle
import os
import sys
from typing import cast

import numpy as np

from .. import inference, llt

logger = logging.getLogger(__name__)


TEST_DIR = os.path.dirname(os.path.realpath(__file__))
FOCUSED = ":" in sys.argv[-1]


class TestInference(unittest.TestCase):
    def test_lltinf_simple(self):
        traces = inference.Traces(
            [
                [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4]],
            ],
            [1, -1],
        )

        lltinf = inference.LLTInf()
        lltinf.fit(traces)

        correct_cut = 4.5
        f = cast(llt.LLTFormula, lltinf.get_formula())
        # logger.debug(lltinf.tree)
        self.assertAlmostEqual(f.pi, correct_cut, places=2)

    def test_lltinf_5_signals(self):
        lltinf = inference.LLTInf()
        traces = inference.Traces(
            [
                [[-1.65955991], [4.40648987], [0.0]],
                [[-2.06465052], [0.77633468], [0.0]],
                [[-1.61610971], [3.70439001], [0.0]],
                [[-1.65390395], [1.17379657], [0.0]],
                [[10.0], [10.0], [0.0]],
            ],
            [-1, 1, -1, -1, 1],
        )

        rho = np.array([3.1377049, 0.99999999, 3.18115509, 1.39746188, 10.22366531])
        tree = lltinf._lltinf(traces, rho, 5, disp=True)
        for i in range(5):
            self.assertEquals(tree.classify(traces.signals[i]), traces.labels[i])

    def test_lltinf_tooclose(self):
        lltinf = inference.LLTInf()
        traces = inference.Traces(
            [
                [[-1.65955991], [4.40648987], [0.0]],
                [[-2.06465052], [0.77633468], [0.0]],
                [[-1.61610971], [3.70439001], [0.0]],
                [[-1.65390395], [1.17379657], [0.0]],
                [[10.0], [-10.0], [0.0]],
            ],
            [-1, 1, -1, -1, 1],
        )

        rho = np.array([3.1377049, 2.73261429, 3.18115509, 3.14336085, 14.7972648])
        tree = lltinf._lltinf(traces, rho, 5, disp=True)
        for i in range(5):
            self.assertEquals(tree.classify(traces.signals[i]), traces.labels[i])

    @unittest.skipUnless(FOCUSED, "Debug test")
    def test_foo(self):
        f = open(os.path.join(TEST_DIR, "debug_tree.pickle"))
        tree = pickle.load(f)
        lltinf = inference.LLTInf(
            0,
            primitive_factory=llt.make_llt_d1_primitives,
            stop_condition=[inference.perfect_stop],
            redo_after_failed=50,
            optimizer_args={"maxiter": 10},
            times=np.arange(0, 5 + 0.0001, 1.0),
        )
        lltinf.tree = tree

        f = 20.62995086001672
        x = np.array(
            [
                [10.0, 10.0, 10.0, 10.0],
                [-8.02971801, -8.02971801, -8.02971801, -8.02971801],
                [5.0, -2.26377207, 0.17309953, -1.80002426],
                [0.0, 1.66666667, 3.33333333, 5.0],
            ]
        )

        traces = inference.Traces([x], [-1])
