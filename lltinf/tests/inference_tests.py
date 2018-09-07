from __future__ import division, absolute_import, print_function

import logging
import unittest

import numpy as np

from lltinf import inference

class TestInference(unittest.TestCase):

    def test_lltinf_simple(self):
        traces = inference.Traces([
            [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
            [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
        ], [1,-1])

        lltinf = inference.LLTInf()
        lltinf.fit(traces)

        print(lltinf.tree)

    def test_lltinf_5_signals(self):
        lltinf = inference.LLTInf()
        traces = inference.Traces([[[-1.65955991],
                [ 4.40648987],
                [ 0.        ]],

                [[-2.06465052],
                [ 0.77633468],
                [ 0.        ]],

                [[-1.61610971],
                [ 3.70439001],
                [ 0.        ]],

                [[-1.65390395],
                [ 1.17379657],
                [ 0.        ]],

                [[10.        ],
                [10.        ],
                [ 0.        ]]],
                [-1, 1, -1, -1, 1])

        rho = np.array([ 3.1377049 ,  0.99999999,  3.18115509,  1.39746188, 10.22366531])
        tree = lltinf._lltinf(traces, rho, 5, disp=True)
        for i in range(5):
            self.assertEquals(tree.classify(traces.signals[i]), traces.labels[i])

    def test_lltinf_tooclose(self):
        lltinf = inference.LLTInf()
        traces = inference.Traces(
            [[[ -1.65955991],
            [  4.40648987],
            [  0.        ]],

        [[ -2.06465052],
            [  0.77633468],
            [  0.        ]],

        [[ -1.61610971],
            [  3.70439001],
            [  0.        ]],

        [[ -1.65390395],
            [  1.17379657],
            [  0.        ]],

        [[ 10.        ],
            [-10.        ],
            [  0.        ]]],
                [-1, 1, -1, -1, 1])

        rho = np.array([ 3.1377049 ,  2.73261429,  3.18115509,  3.14336085, 14.7972648 ])
        tree = lltinf._lltinf(traces, rho, 5, disp=True)
        for i in range(5):
            self.assertEquals(tree.classify(traces.signals[i]), traces.labels[i])
