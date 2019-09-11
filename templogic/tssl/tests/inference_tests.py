import logging
import pickle
import unittest
import os
import sys
from typing import Iterable

import numpy as np
import numpy.testing as npt  # type: ignore

from .. import inference
from templogic import util

logger = logging.getLogger(__name__)
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
        ds_fn = os.path.join(TEST_DIR, "data/spirals.data")
        with open(ds_fn, "rb") as f:
            data = pickle.load(f)
        self.imgs = data["imgs"]
        self.labels = data["labels"]
        self.depth = data["depth"]

    def test_inference_single(self) -> None:
        formula = "((([1]' x - 0.4755859375 < 0)) v ((([1]' x - 0.486328125 < 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{NW} X E_{SW} X ([1]' x - 0.5 > 0) ^ E_{NE} X E_{NE} X E_{NW} X ([1]' x - 0.5625 > 0)) ^ ¬ ([1]' x - 0.4755859375 < 0)) v ((([1]' x - 0.484375 < 0) ^ E_{SE} X E_{NE} X E_{SE} X E_{SE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SE} X E_{NE} X E_{SW} X E_{NE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{SW} X E_{NW} X E_{SE} X E_{NW} X ([1]' x - 0.0 < 0)) ^ ¬ ([1]' x - 0.4755859375 < 0) ^ ¬ (([1]' x - 0.486328125 < 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{NW} X E_{SW} X ([1]' x - 0.5 > 0) ^ E_{NE} X E_{NE} X E_{NW} X ([1]' x - 0.5625 > 0))) v ((([1]' x - 0.4873046875 < 0) ^ E_{NW} X E_{SE} X ([1]' x - 0.484375 < 0) ^ E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 0.75 > 0)) ^ ¬ ([1]' x - 0.4755859375 < 0) ^ ¬ (([1]' x - 0.486328125 < 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{NW} X E_{SW} X ([1]' x - 0.5 > 0) ^ E_{NE} X E_{NE} X E_{NW} X ([1]' x - 0.5625 > 0)) ^ ¬ (([1]' x - 0.484375 < 0) ^ E_{SE} X E_{NE} X E_{SE} X E_{SE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SE} X E_{NE} X E_{SW} X E_{NE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{SW} X E_{NW} X E_{SE} X E_{NW} X ([1]' x - 0.0 < 0))) v ((([1]' x - 0.4873046875 < 0) ^ E_{SE} X E_{SE} X ([1]' x - 0.515625 > 0) ^ E_{NE} X E_{NW} X E_{SE} X E_{NW} X E_{NW} X ([1]' x - 0.0 < 0) ^ E_{SW} X E_{SW} X E_{SW} X E_{NW} X ([1]' x - 0.75 > 0)) ^ ¬ ([1]' x - 0.4755859375 < 0) ^ ¬ (([1]' x - 0.486328125 < 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{NW} X E_{SW} X ([1]' x - 0.5 > 0) ^ E_{NE} X E_{NE} X E_{NW} X ([1]' x - 0.5625 > 0)) ^ ¬ (([1]' x - 0.484375 < 0) ^ E_{SE} X E_{NE} X E_{SE} X E_{SE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SE} X E_{NE} X E_{SW} X E_{NE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{SW} X E_{NW} X E_{SE} X E_{NW} X ([1]' x - 0.0 < 0)) ^ ¬ (([1]' x - 0.4873046875 < 0) ^ E_{NW} X E_{SE} X ([1]' x - 0.484375 < 0) ^ E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 0.75 > 0))) v ((([1]' x - 0.4873046875 < 0) ^ E_{NW} X E_{NE} X E_{SE} X E_{NW} X ([1]' x - 0.25 < 0) ^ E_{SW} X E_{SE} X E_{SW} X E_{SW} X E_{NE} X ([1]' x - 0.0 < 0) ^ E_{NE} X E_{SE} X ([1]' x - 0.5 < 0)) ^ ¬ ([1]' x - 0.4755859375 < 0) ^ ¬ (([1]' x - 0.486328125 < 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{NW} X E_{SW} X ([1]' x - 0.5 > 0) ^ E_{NE} X E_{NE} X E_{NW} X ([1]' x - 0.5625 > 0)) ^ ¬ (([1]' x - 0.484375 < 0) ^ E_{SE} X E_{NE} X E_{SE} X E_{SE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SE} X E_{NE} X E_{SW} X E_{NE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{SW} X E_{NW} X E_{SE} X E_{NW} X ([1]' x - 0.0 < 0)) ^ ¬ (([1]' x - 0.4873046875 < 0) ^ E_{NW} X E_{SE} X ([1]' x - 0.484375 < 0) ^ E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 0.75 > 0)) ^ ¬ (([1]' x - 0.4873046875 < 0) ^ E_{SE} X E_{SE} X ([1]' x - 0.515625 > 0) ^ E_{NE} X E_{NW} X E_{SE} X E_{NW} X E_{NW} X ([1]' x - 0.0 < 0) ^ E_{SW} X E_{SW} X E_{SW} X E_{NW} X ([1]' x - 0.75 > 0))) v ((([1]' x - 0.49609375 < 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{NW} X ([1]' x - 0.25 < 0) ^ E_{SW} X E_{NE} X E_{SW} X E_{NE} X ([1]' x - 0.25 < 0) ^ E_{NE} X ([1]' x - 0.5 < 0) ^ E_{NW} X E_{SW} X E_{NW} X E_{SE} X E_{NE} X ([1]' x - 1.0 > 0) ^ E_{SW} X E_{SW} X E_{NE} X E_{SE} X E_{NE} X ([1]' x - 0.0 < 0)) ^ ¬ ([1]' x - 0.4755859375 < 0) ^ ¬ (([1]' x - 0.486328125 < 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{NW} X E_{SW} X ([1]' x - 0.5 > 0) ^ E_{NE} X E_{NE} X E_{NW} X ([1]' x - 0.5625 > 0)) ^ ¬ (([1]' x - 0.484375 < 0) ^ E_{SE} X E_{NE} X E_{SE} X E_{SE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SE} X E_{NE} X E_{SW} X E_{NE} X E_{NW} X ([1]' x - 1.0 > 0) ^ E_{SW} X E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 1.0 > 0) ^ E_{NW} X E_{SW} X E_{NW} X E_{SE} X E_{NW} X ([1]' x - 0.0 < 0)) ^ ¬ (([1]' x - 0.4873046875 < 0) ^ E_{NW} X E_{SE} X ([1]' x - 0.484375 < 0) ^ E_{NE} X E_{NE} X E_{SE} X E_{SE} X ([1]' x - 0.75 > 0)) ^ ¬ (([1]' x - 0.4873046875 < 0) ^ E_{SE} X E_{SE} X ([1]' x - 0.515625 > 0) ^ E_{NE} X E_{NW} X E_{SE} X E_{NW} X E_{NW} X ([1]' x - 0.0 < 0) ^ E_{SW} X E_{SW} X E_{SW} X E_{NW} X ([1]' x - 0.75 > 0)) ^ ¬ (([1]' x - 0.4873046875 < 0) ^ E_{NW} X E_{NE} X E_{SE} X E_{NW} X ([1]' x - 0.25 < 0) ^ E_{SW} X E_{SE} X E_{SW} X E_{SW} X E_{NE} X ([1]' x - 0.0 < 0) ^ E_{NE} X E_{SE} X ([1]' x - 0.5 < 0))))"
        classifier = inference.TSSLInference()
        dataset = inference.build_dataset(self.imgs, self.labels)
        classifier.build_classifier(dataset, depth=self.depth, valid_class="1")
        logger.debug(str(classifier.get_tssl_formula()))
        self.assertEqual(str(classifier.get_tssl_formula()), formula)

    @unittest.skipUnless(FOCUSED, "Long crossvalidation test")
    def test_inference_cv(self) -> None:
        classifier = inference.TSSLInference()

        class SampleClassifier(util.Classifier):
            def __init__(self, form: inference.tssl.TSSLTerm) -> None:
                self.form = form

            def classify(self, data: np.ndarray) -> int:
                if (
                    inference.tssl.robustness(
                        self.form,
                        inference.tssl.TSSLModel(
                            inference.tssl.QuadTree.from_matrix(data), (2,)
                        ),
                    )
                    >= 0
                ):
                    return 1
                else:
                    return 0

        def build_classifier(
            data: np.ndarray, labels: Iterable[int]
        ) -> util.Classifier:
            dataset = inference.build_dataset(data, labels)
            classifier.build_classifier(dataset, depth=self.depth, valid_class="1")
            return SampleClassifier(classifier.get_tssl_formula())

        cvresult = util.cross_validation(self.imgs, self.labels, build_classifier, k=10)
        self.assertLessEqual(cvresult.miss_mean, 0.2)
