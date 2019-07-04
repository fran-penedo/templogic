from __future__ import division, absolute_import, print_function

import itertools as it
import logging

logger = logging.getLogger(__name__)


class Tree(object):
    def __init__(self, data, children):
        self._data = data
        self.children = children
        self._parent = None

    def get_child(self, idx):
        return self._children[idx]

    def set_child(self, idx, tree):
        if idx >= len(self._children):
            self._children.extend([None for i in range(len(self._children) - idx + 1)])
        self._children[idx] = tree
        tree.parent = self

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, value):
        self._children = value
        for child in self._children:
            child.parent = self

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def level(self):
        if self._parent is None:
            return 0
        else:
            return 1 + self.parent.level()

    def depth(self):
        if self.children is None or all([child is None for child in self._children]):
            return 0
        else:
            return 1 + max([child.depth() for child in self._children])

    def foldl(self, f, z):
        res = z
        l = []
        l.append(self)
        while len(l) > 0:
            t = l.pop(0)
            f(res, t.data)
            l.extend(t.children)

        return res

    def flatten(self):
        return self.foldl(lambda l, a: l.append(a), [])

    def pprint(self, tab=0):
        return _tree_pprint(self, tab)

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return self.__str__()


def _tree_pprint(tree, tab=0):
    pad = " |" * tab + "-"
    if tree is None:
        return pad + "None\n"
    children = [_tree_pprint(child, tab + 1) for child in tree.children]
    return "{}{}\n{}".format(pad, str(tree), "".join(children))
