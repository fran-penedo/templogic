import logging
import unittest
import os

import numpy as np
import numpy.testing as npt

from tssl import tssl, quadtree

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
FOCUSED = os.environ.get("FOCUSED", False)


class TestTSSL(unittest.TestCase):
    def test_formula(self):
        f = tssl.TSSLAnd(
            [
                tssl.TSSLOr(
                    [
                        tssl.TSSLNot(
                            tssl.TSSLExistsNext(
                                [tssl.Direction.NW, tssl.Direction.NE],
                                tssl.TSSLPred([1], 0, tssl.Relation.GE),
                            )
                        ),
                        tssl.TSSLBottom(),
                    ]
                ),
                tssl.TSSLForallNext(
                    [tssl.Direction.SW, tssl.Direction.SE],
                    tssl.TSSLPred([-1], 1, tssl.Relation.LE),
                ),
            ]
        )
