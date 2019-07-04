from __future__ import division, absolute_import, print_function

import itertools as it
import logging
import math

import numpy as np

from tssl.util import Tree

logger = logging.getLogger(__name__)


class QuadTree(Tree):

    """Directions: [0, 1, 2, 3] == [NW, NE, SW, SE]
    """

    def __init__(self, data, children):
        super(QuadTree, self).__init__(data, children)

    @Tree.children.setter
    def children(self, value):
        l = len(value)
        if l != 4 and l != 0:
            raise ValueError("Quad trees must have 4 children per node")
        super(QuadTree, self.__class__).children.fset(self, value)

    def set_child(self, idx, tree):
        if idx >= 4:
            raise ValueError("Quad trees must have 4 children per node")
        else:
            return super(QuadTree, self).set_child(idx, tree)

    def flatten(self):
        flat = super(QuadTree, self).flatten()
        try:
            _ = iter(self.data)
        except TypeError:
            return flat
        else:
            return list(it.chain(*zip(*flat)))

    def isleaf(self):
        return self.children is None or len(self.children) == 0

    @classmethod
    def from_array(cls, x, depth):
        q, r = divmod(len(x), _nnodes(depth))
        if r != 0 or q == 0:
            raise ValueError("Only complete quadtrees allowed")

        if q > 1:
            x = zip(*[x[i * len(x) // q : (i + 1) * len(x) // q] for i in range(q)])

        level = _level(len(x)) - 1
        trees = [[] for i in range(4 ** level)]

        while level > 0:
            data = x[_nnodes(level - 1) : _nnodes(level)]
            trees = [
                [cls(data[4 * i + j], trees[4 * i + j]) for j in range(4)]
                for i in range(len(data) // 4)
            ]
            level -= 1

        return cls(x[0], trees[0])

    @classmethod
    def from_matrix(cls, m, f):
        m = np.array(m)
        depth = int(math.log(m.shape[0], 2))
        if len(m.shape) < 2 or m.shape[0] != m.shape[1] or 2 ** depth != m.shape[0]:
            raise ValueError("Only 2^n x 2^n x d matrices supported")

        if depth == 0:
            return cls(m[0, 0], [])

        children = [
            cls.from_matrix(
                m[
                    i * m.shape[0] // 2 : (i + 1) * m.shape[0] // 2,
                    j * m.shape[0] // 2 : (j + 1) * m.shape[0] // 2,
                ],
                f,
            )
            for i in range(2)
            for j in range(2)
        ]
        return cls(f([c.data for c in children]), children)


def _nnodes(depth):
    if depth < 0:
        return 0
    return (4 ** (depth + 1) - 1) // 3


def _label_to_index(label, depth):
    # if "".join(map(str, label)) == '00001':
    #     import pdb; pdb.set_trace()
    idx = label[0] * _nnodes(depth)
    level = len(label) - 1
    if level == 0:
        return idx
    idx += _nnodes(level - 1)
    for i in label[1:-1]:
        level -= 1
        idx += i * (4 ** level)

    return idx + label[-1]


def _level(idx):
    if idx == 0:
        level = 0
    elif idx == 1:
        level = 1
    else:
        level = int(math.floor(math.log(3 * idx + 1, 4)))
    return level


def _index_to_label(idx, depth):
    # if idx == 341:
    #     import pdb; pdb.set_trace()
    q, idx = divmod(idx, _nnodes(depth))
    label = [q]
    level = _level(idx)
    idx -= _nnodes(level - 1)
    while level > 0:
        level -= 1
        q, idx = divmod(idx, 4 ** level)
        label.append(q)

    return label
