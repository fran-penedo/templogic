import copy
import logging
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Tuple, TypeVar, Iterable, Callable, Sequence, Union

import numpy as np  # type: ignore

from .quadtree import QuadTree

__all__ = [
    "QuadTree",
    "Direction",
    "Relation",
    "TSSLModel",
    "TSSLTerm",
    "DisjTerm",
    "ConjTerm",
    "BooleanTerm",
    "DirectionTerm",
    "TSSLTop",
    "TSSLBottom",
    "TSSLPred",
    "TSSLNot",
    "TSSLOr",
    "TSSLAnd",
    "TSSLExistsNext",
    "TSSLForallNext",
    "robustness",
]


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

    def zoom(self, d: Union[Direction, Iterable[Direction]]) -> "TSSLModel":
        try:
            ds = list(iter(d))  # type: ignore # testing d
        except TypeError:
            ds = [d]  # type: ignore # d tested as Direction
        copy = self.shallowcopy()
        qtree = copy.qtree
        for direction in ds:
            if qtree.isleaf():
                break
            qtree = qtree.children[direction.value]
        copy.qtree = qtree
        return copy

    @property
    def data(self) -> Tuple[float, ...]:
        return self.qtree.data

    def copy(self) -> "TSSLModel":
        return copy.deepcopy(self)

    def shallowcopy(self) -> "TSSLModel":
        return copy.copy(self)


class TSSLTerm(ABC):
    """Any TSSL term is derived from this class

    This class implements general functions to traverse the complete formula and
    delegates the concrete implementation to specific terms
    """

    args: Sequence["TSSLTerm"]

    def __init__(self, args: Sequence["TSSLTerm"]) -> None:
        self.args = args

    def map(self, f: Callable[["TSSLTerm"], "TSSLTerm"]) -> "TSSLTerm":
        args = [arg.map(f) for arg in self.args]
        obj = f(self)
        obj.args = args
        return obj

    def fold(self, g: Callable[["TSSLTerm", Iterable[T]], T]) -> T:
        return g(self, [arg.fold(g) for arg in self.args])

    def directionalFold(
        self,
        g: Callable[["TSSLTerm", Iterable[Direction], Iterable[T]], T],
        dirs: Iterable[Direction],
    ) -> T:
        return g(self, dirs, [arg.directionalFold(g, dirs) for arg in self.args])

    def copy(self) -> "TSSLTerm":
        return copy.deepcopy(self)

    def shallowcopy(self) -> "TSSLTerm":
        return copy.copy(self)

    @abstractmethod
    def __str__(self) -> str:
        pass


class DisjTerm(TSSLTerm):
    pass


class ConjTerm(TSSLTerm):
    pass


class BooleanTerm(TSSLTerm):
    def __init__(self, args: Sequence[TSSLTerm]):
        if len(args) == 0:
            raise ValueError(
                f"{self.__class__.__name__} must have at least one argument"
            )
        super().__init__(args)


class DirectionTerm(TSSLTerm):
    dirs: Iterable[Direction]

    def __init__(self, dirs: Iterable[Direction], arg: TSSLTerm):
        super().__init__([arg])
        self.dirs = dirs

    def directionalFold(
        self,
        g: Callable[["TSSLTerm", Iterable[Direction], Iterable[T]], T],
        dirs: Iterable[Direction],
    ) -> T:
        return g(
            self,
            dirs,
            [self.args[0].directionalFold(g, list(dirs) + [d]) for d in self.dirs],
        )


class TSSLTop(ConjTerm, TSSLTerm):
    def __init__(self):
        super().__init__([])

    def __str__(self) -> str:
        return "T"


class TSSLBottom(ConjTerm, TSSLTerm):
    def __init__(self):
        super().__init__([])

    def __str__(self) -> str:
        return "_|_"


class TSSLPred(ConjTerm, TSSLTerm):

    """Predicate of the form a' * x - b ~ 0
    """

    a: Sequence[float]
    b: float
    rel: Relation

    def __init__(self, a: Sequence[float], b: float, rel: Relation):
        super().__init__([])
        self.a = a
        self.b = b
        self.rel = rel

    def __str__(self):
        return "({}' x - {} {} 0)".format(str(self.a), str(self.b), str(self.rel))


class TSSLNot(ConjTerm, BooleanTerm, TSSLTerm):
    def __init__(self, arg: TSSLTerm):
        super().__init__([arg])

    def __str__(self) -> str:
        return "¬ {}".format(str(self.args[0]))


class TSSLAnd(BooleanTerm, ConjTerm, TSSLTerm):
    def __str__(self) -> str:
        return "({})".format(" ^ ".join(str(arg) for arg in self.args))


class TSSLOr(BooleanTerm, DisjTerm, TSSLTerm):
    def __str__(self) -> str:
        return "({})".format(" v ".join(str(arg) for arg in self.args))


class TSSLExistsNext(DirectionTerm, DisjTerm, TSSLTerm):
    def __str__(self) -> str:
        return "E_{{{}}} X {}".format(
            ",".join(str(d) for d in self.dirs), str(self.args[0])
        )


class TSSLForallNext(DirectionTerm, ConjTerm, TSSLTerm):
    def __str__(self) -> str:
        return "A_{{{}}} X {}".format(
            ",".join(str(d) for d in self.dirs), str(self.args[0])
        )


def robustness(formula: TSSLTerm, model: TSSLModel) -> float:
    def _rob(term: TSSLTerm, dirs: Iterable[Direction], robs: Iterable[float]) -> float:
        if isinstance(term, DirectionTerm):
            robs = [0.25 * r for r in robs]  # FIXME might not be correct
        if isinstance(term, TSSLTop):
            return min(model.bound)
        elif isinstance(term, TSSLBottom):
            return -max(model.bound)
        elif isinstance(term, TSSLPred):
            zoomed_model = model.zoom(dirs)
            dif = len(zoomed_model.data) - len(term.a)
            if dif > 0:
                a = np.pad(term.a, [(0, dif)], mode="constant")
            else:
                a = term.a
            return term.rel.value * (np.dot(a, zoomed_model.data) - term.b)
        elif isinstance(term, TSSLNot):
            return -list(robs)[0]
        elif isinstance(term, ConjTerm):
            return min(robs)
        elif isinstance(term, DisjTerm):
            return max(robs)
        else:
            raise Exception(f"Non exhaustive pattern matching {term.__class__}")

    return formula.directionalFold(_rob, [])
