import unittest

import numpy as np

from .. import spatel

import logging

logger = logging.getLogger(__name__)


class TestSpatel(unittest.TestCase):
    def setUp(self) -> None:
        self.tssl_f = spatel.TSSLAnd(
            [
                spatel.TSSLOr(
                    [
                        spatel.TSSLNot(
                            spatel.TSSLExistsNext(
                                [spatel.Direction.NW, spatel.Direction.NE],
                                spatel.TSSLPred([1], 0, spatel.Relation.GE),
                            )
                        ),
                        spatel.TSSLBottom(),
                    ]
                ),
                spatel.TSSLForallNext(
                    [spatel.Direction.SW, spatel.Direction.SE],
                    spatel.TSSLPred([-1], 1, spatel.Relation.LE),
                ),
                spatel.TSSLTop(),
            ]
        )

        self.f = spatel.STLNext(
            spatel.STLAlways(
                bounds=(3, 5),
                arg=spatel.STLEventually(
                    bounds=(2, 4),
                    arg=spatel.STLAnd(
                        args=[
                            spatel.STLOr(
                                args=[
                                    spatel.SpatelSTLPred(self.tssl_f),
                                    spatel.STLAlways(
                                        bounds=(1, 3),
                                        arg=spatel.SpatelSTLPred(self.tssl_f),
                                    ),
                                    spatel.STLEventually(
                                        bounds=(0, 2),
                                        arg=spatel.SpatelSTLPred(self.tssl_f),
                                    ),
                                ]
                            ),
                            spatel.STLNot(spatel.SpatelSTLPred(self.tssl_f)),
                        ]
                    ),
                ),
            )
        )

    def test_robustness(self) -> None:
        qts = [
            spatel.QuadTree.from_matrix(
                [[[t + 0], [t + 1]], [[t + 2], [t + 3]]], np.mean
            )
            for t in range(20)
        ]

        model = spatel.SpatelModel(qts, (24,))

        self.assertEqual(spatel.robustness(self.f, model, 0), -2.25)
