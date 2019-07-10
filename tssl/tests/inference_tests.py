import logging
import unittest
import os

import numpy as np  # type: ignore
import numpy.testing as npt  # type: ignore

from tssl import inference

LOGGER = logging.getLogger(__name__)
TEST_DIR = os.path.dirname(os.path.realpath(__file__))
FOCUSED = os.environ.get("FOCUSED", False)


def setUpModule():
    inference.start_jvm()


def tearDownModule():
    inference.stop_jvm()


class TestInference(unittest.TestCase):
    def setUp(self):
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

    def test_build_dataset(self):
        dataset = inference.build_dataset(self.imgs, self.labels1)
        for i in range(5):
            self.assertEqual(dataset.attribute(i).name, "x{}".format(i))
        self.assertTrue(dataset.attribute(5).jwrapper.isNominal())
        labels = [inst.get_value(5) for inst in dataset]
        npt.assert_array_equal(labels, self.labels1)

    def test_inference(self):
        dataset = inference.build_dataset(self.imgs, self.labels2)
        classifier = inference.TSSLInference()
        classifier.build_classifier(dataset)
        self.assertEqual(str(classifier.get_tssl_formula()), "")
