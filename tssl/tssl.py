import copy
import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Tuple, TypeVar, Iterable, Callable, Sequence
from typing_extensions import Protocol

import numpy as np  # type: ignore

from tssl.quadtree import QuadTree


logger = logging.getLogger(__name__)

T = TypeVar("T")


class Direction(IntEnum):
    """TSSL Directions

    Int value represents index in arrays
    """

    NW = 0
    NE = 1
    SW = 2
    SE = 3

    def __str__(self) -> str:
        return self.name


class Relation(IntEnum):
    """TSSL predicate relations

    Int value represents multiplier in robustness
    """

    LE = -1
    GE = 1

    def __str__(self) -> str:
        return "<" if self is Relation.LE else ">"


class TSSLModel(object):
    """Model for TSSL logic
    """

    qtree: QuadTree[Tuple[float, ...]]
    bound: Tuple[float, ...]

    def __init__(
        self, qtree: QuadTree[Tuple[float, ...]], bound: Tuple[float, ...]
    ) -> None:
        self.qtree = qtree
        self.bound = bound

    def zoom(self, d: Direction) -> "TSSLModel":
        copy = self.copy()
        if not copy.qtree.isleaf():
            copy.qtree = copy.qtree.children[d.value]
        return copy

    @property
    def data(self) -> Tuple[float, ...]:
        return self.qtree.data

    def copy(self) -> "TSSLModel":
        return copy.copy(self)


class MMap(Protocol[T]):
    def __call__(
        self,
        model: TSSLModel,
        mmap: Callable[["TSSLTerm"], "MMap[T]"],
        mreduce: Callable[["TSSLTerm"], "MReduce[T]"],
    ) -> Iterable[T]:
        pass


class MReduce(Protocol[T]):
    def __call__(self, rhos: Iterable[T]) -> T:
        pass


MMapGet = Callable[["TSSLTerm"], MMap[T]]
MReduceGet = Callable[["TSSLTerm"], MReduce[T]]


class TSSLTerm(ABC):
    """Any TSSL term is derived from this class

    This class implements general functions to traverse the complete formula and
    delegates the concrete implementation to specific terms
    """

    def score(self, model: TSSLModel, mmap: MMapGet[T], mreduce: MReduceGet[T]) -> T:
        return mreduce(self)(mmap(self)(model, mmap, mreduce))

    def robustness(self, model: TSSLModel) -> float:
        return self.score(model, lambda obj: obj.rho_map, lambda obj: obj.rho_reduce)

    @abstractmethod
    def rho_map(
        self, model: TSSLModel, mmap: MMapGet[float], mreduce: MReduceGet[float]
    ) -> Iterable[float]:
        pass

    @abstractmethod
    def rho_reduce(self, rhos: Iterable[float]) -> float:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


class DisjTerm(object):
    def rho_reduce(self, rhos: Iterable[float]) -> float:
        return max(rhos)


class ConjTerm(object):
    def rho_reduce(self, rhos: Iterable[float]) -> float:
        return min(rhos)


class BooleanTerm(object):
    args: Sequence[TSSLTerm]

    def rho_map(
        self, model: TSSLModel, mmap: MMapGet[float], mreduce: MReduceGet[float]
    ) -> Iterable[float]:
        return [arg.score(model, mmap, mreduce) for arg in self.args]


class DirectionTerm(object):
    arg: TSSLTerm
    dirs: Iterable[Direction]

    def rho_map(
        self, model: TSSLModel, mmap: MMapGet[float], mreduce: MReduceGet[float]
    ) -> Iterable[float]:
        # FIXME the .25 only makes sense for mean-based quadtrees
        return [0.25 * self.arg.score(model.zoom(d), mmap, mreduce) for d in self.dirs]


class TSSLTop(ConjTerm, TSSLTerm):
    def rho_map(
        self, model: TSSLModel, mmap: MMapGet[float], mreduce: MReduceGet[float]
    ) -> Iterable[float]:
        return [min(model.bound)]

    def __str__(self) -> str:
        return "T"


class TSSLBottom(ConjTerm, TSSLTerm):
    def rho_map(
        self, model: TSSLModel, mmap: MMapGet[float], mreduce: MReduceGet[float]
    ) -> Iterable[float]:
        return [-max(model.bound)]

    def __str__(self) -> str:
        return "_|_"


class TSSLPred(ConjTerm, TSSLTerm):

    """Predicate of the form a' * x - b ~ 0
    """

    a: Sequence[float]
    b: float
    rel: Relation

    def __init__(self, a: Sequence[float], b: float, rel: Relation):
        self.a = a
        self.b = b
        self.rel = rel

    def rho_map(
        self, model: TSSLModel, mmap: MMapGet[float], mreduce: MReduceGet[float]
    ) -> Iterable[float]:
        dif = len(model.data) - len(self.a)
        if dif > 0:
            a = np.pad(self.a, [(0, dif)], mode="constant")
        else:
            a = self.a
        return [self.rel.value * (np.dot(a, model.data) - self.b)]

    def __str__(self):
        return "({}' x - {} {} 0)".format(str(self.a), str(self.b), str(self.rel))


class TSSLNot(ConjTerm, BooleanTerm, TSSLTerm):
    def __init__(self, arg: TSSLTerm):
        self.args: Sequence[TSSLTerm] = [arg]

    def rho_reduce(self, rhos: Iterable[float]) -> float:
        return -super(TSSLNot, self).rho_reduce(rhos)

    def __str__(self) -> str:
        return "¬ {}".format(str(self.args[0]))


class TSSLAnd(BooleanTerm, ConjTerm, TSSLTerm):
    def __init__(self, args: Sequence[TSSLTerm]):
        if len(args) == 0:
            raise ValueError("TSSLAnd must have at least one argument")
        self.args = args

    def __str__(self) -> str:
        return "({})".format(" ^ ".join(str(arg) for arg in self.args))


class TSSLOr(BooleanTerm, DisjTerm, TSSLTerm):
    def __init__(self, args: Sequence[TSSLTerm]):
        if len(args) == 0:
            raise ValueError("TSSLOr must have at least one argument")
        self.args = args

    def __str__(self) -> str:
        return "({})".format(" v ".join(str(arg) for arg in self.args))


class TSSLExistsNext(DirectionTerm, DisjTerm, TSSLTerm):
    def __init__(self, dirs: Iterable[Direction], arg: TSSLTerm):
        self.arg = arg
        self.dirs = dirs

    def __str__(self) -> str:
        return "E_{{{}}} X {}".format(
            ",".join(str(d) for d in self.dirs), str(self.arg)
        )


class TSSLForallNext(DirectionTerm, ConjTerm, TSSLTerm):
    def __init__(self, dirs: Iterable[Direction], arg: TSSLTerm):
        self.arg = arg
        self.dirs = dirs

    def __str__(self) -> str:
        return "A_{{{}}} X {}".format(
            ",".join(str(d) for d in self.dirs), str(self.arg)
        )
