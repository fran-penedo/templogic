import unittest

import numpy as np  # type: ignore
import numpy.testing as npt  # type: ignore

import logging

logger = logging.getLogger(__name__)

from .. import util


class TestUtil(unittest.TestCase):
    def test_split_groups(self) -> None:
        x = [1, -1, 2, -2]
        p, n = util.split_groups(x, lambda t: t >= 0)
        npt.assert_array_equal(p, [1, 2])
        npt.assert_array_equal(n, [-1, -2])
