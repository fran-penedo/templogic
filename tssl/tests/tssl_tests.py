import logging
import unittest
import os

import numpy as np  # type: ignore
import numpy.testing as npt  # type: ignore

from tssl import tssl, quadtree

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
FOCUSED = os.environ.get("FOCUSED", False)


class TestTSSL(unittest.TestCase):
    def setUp(self) -> None:
        self.f = tssl.TSSLAnd(
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
                tssl.TSSLTop(),
            ]
        )

    def test_formula_str(self) -> None:
        s = "((¬ E_{NW,NE} X ([1]' x - 0 > 0) v _|_) ^ A_{SW,SE} X ([-1]' x - 1 < 0) ^ T)"
        npt.assert_equal(str(self.f), s)

    def test_robustness(self) -> None:
        qt = quadtree.QuadTree.from_matrix([[0, 1], [2, 3]], np.mean)
        model = tssl.TSSLModel(qt, (4,))
        npt.assert_equal(self.f.robustness(model), -0.25)
        npt.assert_equal(self.f.args[1].robustness(model), 0.75)
