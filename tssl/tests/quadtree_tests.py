from __future__ import division, absolute_import, print_function

import logging
import unittest
import os
from functools import partial

import numpy as np
import numpy.testing as npt

from tssl import quadtree

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
FOCUSED = os.environ.get("FOCUSED", False)


class TestQuadTree(unittest.TestCase):
    def test_labels(self):
        for i in range(8):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        for m in range(4):
                            label = []
                            for d in [i, j, k, l, m]:
                                label.append(d)
                                idx = quadtree._label_to_index(label, 5)
                                label2 = quadtree._index_to_label(idx, 5)
                                npt.assert_array_equal(label, label2)

        for idx in range(200):
            label = quadtree._index_to_label(idx, 3)
            self.assertEqual(idx, quadtree._label_to_index(label, 3))

    def test_flatten(self):
        array = list(range(quadtree._nnodes(5)))
        tree = quadtree.QuadTree.from_array(array, 5)
        array2 = tree.flatten()
        npt.assert_array_equal(array, array2)

    def test_flatten2(self):
        array = list(range(3 * quadtree._nnodes(5)))
        tree = quadtree.QuadTree.from_array(array, 5)
        array2 = tree.flatten()
        npt.assert_array_equal(array, array2)
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        for m in range(4):
                            for n in range(4):
                                label = []
                                for d in [i, j, k, l, m, n]:
                                    label.append(d)
                                    idx = quadtree._label_to_index(label, 5)
                                    self.assertEqual(array[idx], idx)

    def test_from_matrix(self):
        m = np.array([[[0, 4], [1, 5]], [[2, 6], [3, 7]]])
        res = np.array([1.5, 0, 1, 2, 3, 5.5, 4, 5, 6, 7])
        tree = quadtree.QuadTree.from_matrix(m, partial(np.mean, axis=0))
        npt.assert_array_equal(tree.flatten(), res)
