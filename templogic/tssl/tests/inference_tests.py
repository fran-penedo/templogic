import logging
import unittest
import os
import sys

import numpy as np  # type: ignore
import numpy.testing as npt  # type: ignore

from .. import inference

LOGGER = logging.getLogger(__name__)
TEST_DIR = os.path.dirname(os.path.realpath(__file__))
FOCUSED = ":" in sys.argv[-1]


def setUpModule():
    inference.start_jvm()


def tearDownModule():
    inference.stop_jvm()


class TestInference(unittest.TestCase):
    def setUp(self) -> None:
        self.imgs = [
            [[[1], [1]], [[0], [0]]],
            [[[1], [0]], [[0], [0]]],
            [[[0], [1]], [[0], [0]]],
            [[[0], [0]], [[1], [0]]],
            [[[1], [1]], [[0], [0]]],
            [[[1], [0]], [[0], [0]]],
            [[[0], [1]], [[0], [0]]],
            [[[0], [0]], [[1], [0]]],
            [[[1], [1]], [[0], [0]]],
            [[[1], [0]], [[0], [0]]],
            [[[0], [1]], [[0], [0]]],
            [[[0], [0]], [[1], [0]]],
            [[[1], [1]], [[0], [0]]],
            [[[1], [0]], [[0], [0]]],
            [[[0], [1]], [[0], [0]]],
            [[[0], [0]], [[1], [0]]],
        ]
        self.labels1 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        self.labels2 = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]

    def test_build_dataset(self) -> None:
        dataset = inference.build_dataset(self.imgs, self.labels1)
        for i in range(5):
            self.assertEqual(dataset.attribute(i).name, "x{}".format(i))
        self.assertTrue(dataset.attribute(5).jwrapper.isNominal())
        labels = [inst.get_value(5) for inst in dataset]
        npt.assert_array_equal(labels, self.labels1)

    def test_inference(self) -> None:
        formulas = [
            "(¬ ([1]' x - 0.5 > 0))",
            "((([1]' x - 0.5 > 0)) v (E_{SW} X ([1]' x - 1.0 > 0) ^ ¬ ([1]' x - 0.5 > 0)))",
        ]
        for i, labels in enumerate([self.labels1, self.labels2]):
            dataset = inference.build_dataset(self.imgs, labels)
            classifier = inference.TSSLInference()
            classifier.build_classifier(dataset, depth=1)
            self.assertEqual(str(classifier.get_tssl_formula()), formulas[i])


class TestSpiralDataSet(unittest.TestCase):
    def setUp(self) -> None:
        pass
