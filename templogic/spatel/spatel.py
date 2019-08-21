"""Module with SpaTeL definitions

"""

import logging
from typing import Iterable, Sequence, Tuple, TypeVar

from templogic import stlmilp as stl, tssl
from templogic.stlmilp import STLAlways, STLAnd, STLEventually, STLNext, STLNot, STLOr
from templogic.tssl import (
    Direction,
    Relation,
    TSSLBottom,
    TSSLExistsNext,
    TSSLForallNext,
    TSSLNot,
    TSSLOr,
    TSSLAnd,
    TSSLPred,
    TSSLTop,
    QuadTree,
)

__all__ = [
    "STLAlways",
    "STLAnd",
    "STLEventually",
    "STLNext",
    "STLNot",
    "STLOr",
    "Direction",
    "Relation",
    "TSSLBottom",
    "TSSLExistsNext",
    "TSSLForallNext",
    "TSSLNot",
    "TSSLOr",
    "TSSLAnd",
    "TSSLPred",
    "TSSLTop",
    "SpatelModel",
    "SpatelSTLPred",
    "robustness",
    "QuadTree",
]


logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

SpatelTerm = stl.STLTerm


class SpatelModel(stl.STLModel[int, tssl.TSSLModel]):
    tssl_models: Sequence[tssl.TSSLModel]

    def __init__(
        self, qtrees: Iterable[tssl.QuadTree], bound: Tuple[float, ...]
    ) -> None:
        self.tinter = 1
        self.tssl_models = [tssl.TSSLModel(qtree, bound) for qtree in qtrees]

    def getVarByName(self, label_t: int) -> tssl.TSSLModel:
        return self.tssl_models[label_t]


class TSSLTermSignal(stl.Signal[tssl.TSSLModel, int]):
    def __init__(self, term: tssl.TSSLTerm) -> None:
        super().__init__(
            lambda t: (int(t),), lambda vs: tssl.robustness(self.term, vs[0])
        )
        self.term = term

        def __str__(self):
            return str(self.term)


class SpatelSTLPred(stl.STLPred):
    def __init__(self, tssl_term: tssl.TSSLTerm) -> None:
        super().__init__(TSSLTermSignal(tssl_term))


def robustness(term: SpatelTerm, model: SpatelModel, t: float = 0) -> float:
    return stl.robustness(term, model, t)
