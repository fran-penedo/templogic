import copy
import logging
from abc import ABC, abstractmethod
from enum import IntEnum

import numpy as np  # type: ignore


logger = logging.getLogger(__name__)


class Direction(IntEnum):
    NW = 0
    NE = 1
    SW = 2
    SE = 3

    def __str__(self):
        return self.name


class Relation(IntEnum):
    LE = -1
    GE = 1

    def __str__(self):
        return "<" if self is Relation.LE else ">"


class TSSLModel(object):
    def __init__(self, qtree, bound):
        self.qtree = qtree
        self.bound = bound

    def zoom(self, d):
        copy = self.copy()
        if not copy.qtree.isleaf():
            copy.qtree = copy.qtree.children[d.value]
        return copy

    @property
    def data(self):
        return self.qtree.data

    def copy(self):
        return copy.copy(self)


class TSSLTerm(ABC):
    def score(self, model, mmap, mreduce):
        return mreduce(self)(mmap(self)(model, mmap, mreduce))

    def robustness(self, model: TSSLModel) -> float:
        return self.score(model, lambda obj: obj.rho_map, lambda obj: obj.rho_reduce)

    @abstractmethod
    def rho_map(self, model, mmap, mreduce):
        pass

    @abstractmethod
    def rho_reduce(self, rhos):
        pass

    @abstractmethod
    def __str__(self):
        pass


class TSSLTop(TSSLTerm):
    def rho_map(self, model, mmap, mreduce):
        return [model.bound]

    def rho_reduce(self, rhos):
        return rhos[0]

    def __str__(self):
        return "T"


class TSSLBottom(TSSLTerm):
    def rho_map(self, model, mmap, mreduce):
        return [-model.bound]

    def rho_reduce(self, rhos):
        return rhos[0]

    def __str__(self):
        return "_|_"


class TSSLPred(TSSLTerm):

    """Predicate of the form a' * x - b ~ 0
    """

    def __init__(self, a, b, rel):
        self._a = a
        self._b = b
        self._rel = rel

    def rho_reduce(self, rhos):
        return rhos[0]

    def rho_map(self, model, mmap, mreduce):
        return [self._rel.value * (np.dot(self._a, model.data) - self._b)]

    def __str__(self):
        return "({}' x - {} {} 0)".format(str(self._a), str(self._b), str(self._rel))


class TSSLNot(TSSLTerm):
    def __init__(self, arg):
        self._arg = arg

    def rho_reduce(self, rhos):
        return -rhos[0]

    def rho_map(self, model, mmap, mreduce):
        return [self._arg.score(model, mmap, mreduce)]

    def __str__(self):
        return "¬ {}".format(str(self._arg))


def _boolean_map(args, model, mmap, mreduce):
    return [arg.score(model, mmap, mreduce) for arg in args]


class TSSLAnd(TSSLTerm):
    def __init__(self, args):
        if len(args) == 0:
            raise ValueError("TSSLAnd must have at least one argument")
        self._args = args

    def rho_reduce(self, rhos):
        return min(rhos)

    def rho_map(self, model, mmap, mreduce):
        return _boolean_map(self._args, model, mmap, mreduce)

    def __str__(self):
        return "({})".format(" ^ ".join(str(arg) for arg in self._args))


class TSSLOr(TSSLTerm):
    def __init__(self, args):
        if len(args) == 0:
            raise ValueError("TSSLOr must have at least one argument")
        self._args = args

    def rho_reduce(self, rhos):
        return max(rhos)

    def rho_map(self, model, mmap, mreduce):
        return _boolean_map(self._args, model, mmap, mreduce)

    def __str__(self):
        return "({})".format(" v ".join(str(arg) for arg in self._args))


def _dir_map(arg, dirs, model, mmap, mreduce):
    # FIXME the .25 only makes sense for mean-based quadtrees
    return [0.25 * arg.score(model.zoom(d), mmap, mreduce) for d in dirs]


class TSSLExistsNext(TSSLTerm):
    def __init__(self, dirs, arg):
        self._arg = arg
        self._dirs = dirs

    def rho_reduce(self, rhos):
        return max(rhos)

    def rho_map(self, model, mmap, mreduce):
        return _dir_map(self._arg, self._dirs, model, mmap, mreduce)

    def __str__(self):
        return "E_{{{}}} X {}".format(
            ",".join(str(d) for d in self._dirs), str(self._arg)
        )


class TSSLForallNext(TSSLTerm):
    def __init__(self, dirs, arg):
        self._arg = arg
        self._dirs = dirs

    def rho_reduce(self, rhos):
        return min(rhos)

    def rho_map(self, model, mmap, mreduce):
        return _dir_map(self._arg, self._dirs, model, mmap, mreduce)

    def __str__(self):
        return "A_{{{}}} X {}".format(
            ",".join(str(d) for d in self._dirs), str(self._arg)
        )
