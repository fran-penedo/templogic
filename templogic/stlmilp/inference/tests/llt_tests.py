from __future__ import division, absolute_import, print_function

import unittest
import pprint

import numpy as np
import numpy.testing as npt  # type: ignore

import logging

logger = logging.getLogger(__name__)

from .. import llt


class TestLLT(unittest.TestCase):
    def test_primitives(self) -> None:
        signals = [
            [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
            [[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 4]],
        ]
        prims = list(llt.make_llt_primitives(signals))
        self.assertEqual(len(prims), 4)
        prim = prims[0]
        prim.set_llt_pars((1, 2, 3, 4))
        self.assertEqual(prim.t0, 1)
        self.assertEqual(prim.t1, 2)
        self.assertEqual(prim.t3, 3)
        self.assertEqual(prim.pi, 4)
        prim = prim.copy()
        self.assertEqual(prim.t0, 1)
        self.assertEqual(prim.t1, 2)
        self.assertEqual(prim.t3, 3)
        self.assertEqual(prim.pi, 4)
