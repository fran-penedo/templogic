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


class TestSpiralDataSet(unittest.TestCase):
    def setUp(self) -> None:
        ds_fn = os.path.join(TEST_DIR, "data/spirals.data")
        with open(ds_fn, "rb") as f:
            data = pickle.load(f)
        self.imgs = data["signals"]
        self.labels = data["labels"]
        self.depth = data["depth"]

    def test_inference_single(self) -> None:
        formula = "foobar"
        classifier = inference.SpatelInference(log=True)
        imgs = self.imgs[:100] + self.imgs[-100:]
        labels = np.append(self.labels[:100], self.labels[-100:])
        dataset = inference.SpatelTraces.from_matrices(imgs, labels)
        classifier.fit(dataset)
        logger.debug(str(classifier.get_formula()))
        self.assertEqual(str(classifier.get_formula()), formula)

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
            classifier.build_classifier(dataset, valid_class="1")
            return SampleClassifier(classifier.get_tssl_formula())

        cvresult = util.cross_validation(self.imgs, self.labels, build_classifier, k=10)
        self.assertLessEqual(cvresult.miss_mean, 0.2)
