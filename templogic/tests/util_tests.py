import unittest

import numpy as np
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

    def test_missrate(self) -> None:
        data = list(range(5))
        labels = [1, 1, 1, -1, -1]

        class SampleClassifier(util.Classifier):
            def classify(self, x):
                return 1 if x < 2 else -1

        npt.assert_almost_equal(
            util.missrate(data, labels, SampleClassifier()), 1.0 / 5
        )

    def test_cv(self) -> None:
        data = list(range(10))
        labels = [1, 1, 1, -1, -1, 1, -1, 1, 1, 1]

        class SampleClassifier(util.Classifier):
            def classify(self, x):
                return 1 if x < 2 else -1

        def build_classifier(data, labels):
            return SampleClassifier()

        self.assertEquals(
            util.cross_validation(data, labels, build_classifier, k=5).miss_mean, 0.5
        )
