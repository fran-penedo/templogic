"""Module with depth 2 p-stl definitions

Author: Francisco Penedo (franp@bu.edu)

"""
from . import stl
import itertools
from enum import Enum
import operator
from typing import Tuple, Iterable, Callable, Sequence, Union, cast

import numpy as np  # type: ignore
from pyparsing import Word, alphas, Suppress, nums, Literal, MatchFirst  # type: ignore


class Relation(Enum):
    LE = operator.le
    GT = operator.ge

    @classmethod
    def from_str(cls, s: str) -> "Relation":
        if s == "<" or s == "<=":
            return cls.LE
        elif s == ">" or s == ">=":
            return cls.GT
        else:
            raise ValueError(f"String '{s}' cannot be converted to a relation")

    def flip(self) -> "Relation":
        if self == Relation.LE:
            return Relation.GT
        else:
            return Relation.LE

    def __str__(self):
        return "<=" if self == Relation.LE else ">"


class LLTSignal(stl.Signal):
    """Definition of an atomic proposition in LLT: x_j ~ pi, where ~ is <= or >
    """

    _index: int
    _op: Relation
    _pi: float

    def __init__(
        self, index: int = 0, op: Union[str, Relation] = "<", pi: float = 0
    ) -> None:
        """Creates a signal x_j ~ pi.

        index : integer
                Corresponds to j.
        op : either "<" or ">"
             Corresponds to op.
        pi : numeric

        """
        self.index = index
        self.op = op
        self.pi = pi
        self._setup()

    def _setup(self) -> None:
        # labels transform a time into a pair [j, t]
        self.labels = [lambda t: (self._index, t)]
        # transform to x_j - pi >= 0
        self.f = lambda vs: (vs[0] - self._pi) * (-1 if self._op == Relation.LE else 1)

    def __deepcopy__(self, memo):
        return LLTSignal(self.index, self.op, self.pi)

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict["_f"]
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)
        self._setup()

    def __str__(self) -> str:
        return f"x_{self.index} {self.op} {self.pi:.2f}"

    @property
    def pi(self) -> float:
        return self._pi

    @pi.setter
    def pi(self, value: float) -> None:
        self._pi = value

    @property
    def op(self) -> Relation:
        return self._op

    @op.setter
    def op(self, value: Union[Relation, str]) -> None:
        self._op = Relation.from_str(value) if isinstance(value, str) else value

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int) -> None:
        self._index = value


class LLTFormula(stl.STLAnd):
    """A depth 2 STL formula.
    """

    signal: LLTSignal
    inner: stl.TemporalTerm
    outer: stl.TemporalTerm

    def __init__(self, live: bool, index: int, op: Union[str, Relation]) -> None:
        """Creates a depth 2 STL formula.

        live : boolean
               True if the formula has a liveness structure (always eventually)
               or not (eventually always).
        index : integer
                Index for the signal (see LLTSignal)
        op : either LE or GT
             Operator for the atomic proposition (see LLTSignal)
        """
        self.signal = LLTSignal(index, op)
        if live:
            self.inner = stl.STLEventually((0, 0), stl.STLPred(self.signal))
            self.outer = stl.STLAlways((0, 0), self.inner)
        else:
            self.inner = stl.STLAlways((0, 0), stl.STLPred(self.signal))
            self.outer = stl.STLEventually((0, 0), self.inner)
        super().__init__([self.outer])

    @property
    def index(self) -> int:
        return self.signal.index

    @property
    def pi(self) -> float:
        return self.signal.pi

    @pi.setter
    def pi(self, value: float) -> None:
        self.signal.pi = value

    @property
    def op(self) -> Relation:
        return self.signal.op

    @op.setter
    def op(self, value: Union[str, Relation]) -> None:
        self.signal.op = value  # type: ignore # Incorrectly handling get/set?

    @property
    def t0(self) -> float:
        return self.outer.bounds[0]

    @t0.setter
    def t0(self, value: float) -> None:
        self.outer.bounds = (value, self.outer.bounds[1])

    @property
    def t1(self) -> float:
        return self.outer.bounds[1]

    @t1.setter
    def t1(self, value: float) -> None:
        self.outer.bounds = (self.outer.bounds[0], value)

    @property
    def t3(self) -> float:
        return self.inner.bounds[1]

    @t3.setter
    def t3(self, value: float) -> None:
        self.inner.bounds = (self.inner.bounds[0], value)

    def reverse_op(self) -> None:
        """Reverses the operator of the atomic proposition
        """
        self.op = self.op.flip()

    def copy(self) -> "LLTFormula":
        return cast("LLTFormula", super().copy())

    def parameter_bounds(self, maxt: float, minpi: float, maxpi: float) -> Tuple:
        return [0, 0, minpi], [maxt, maxt, maxpi]


def set_llt_pars(
    primitive: LLTFormula, t0: float, t1: float, t3: float, pi: float
) -> None:
    """Sets the parameters of a primitive
    """
    primitive.t0 = t0
    primitive.t1 = t1
    primitive.t3 = t3
    primitive.pi = pi


SignalType = Sequence[Sequence[float]]


class SimpleModel(stl.STLModel):
    """Matrix-like model with fixed sample interval.

    Parameters
    ----------
    signal : m by n matrix.
        Last row should be the sampling times.

    """

    _signal: SignalType
    _lsignal: int

    def __init__(self, signal: SignalType) -> None:
        self._signal = signal
        try:
            self.tinter = signal[-1][1] - signal[-1][0]
        except IndexError:  # only one time
            self.tinter = 1
        self._lsignal = len(signal[-1])

    def getVarByName(self, indices: Tuple[int, float]) -> float:
        """ Get variables

        indices : pair of numerics
                  indices[0] represents the name of the signal
                  indices[1] represents the time at which to sample the signal

        FIXME: Assumes that sampling rate is constant, i.e. the sampling
        times are in arithmetic progression with rate self._tinter
        """
        tindex = int(min(np.floor(indices[1] / self.tinter), self._lsignal - 1))
        # assert 0 <= tindex <= len(self._signals[-1]), \
        #        'Invalid query outside the time domain of the trace! %f' % tindex
        return self._signal[indices[0]][tindex]


def make_llt_primitives(signals: Sequence[SignalType]) -> Iterable[LLTFormula]:
    """Obtains the depth 2 primitives associated with the structure of the signals.

    signals : m by n matrix
              Last column should be the sampling times
    """
    alw_ev = [
        LLTFormula(True, index, op)
        for index, op in itertools.product(range(len(signals[0]) - 1), [Relation.LE])
    ]
    ev_alw = [
        LLTFormula(False, index, op)
        for index, op in itertools.product(range(len(signals[0]) - 1), [Relation.LE])
    ]

    return alw_ev + ev_alw


def expr_parser():
    num = stl.num_parser()

    T_UND = Suppress(Literal("_"))
    T_LE = Literal("<=")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    relation = (T_LE | T_GR).setParseAction(
        lambda t: Relation.LE if t[0] == "<=" else Relation.GT
    )
    expr = Suppress(Word(alphas)) + T_UND + integer + relation + num
    expr.setParseAction(lambda t: LLTSignal(t[0], t[1], t[2]))

    return expr


def llt_parser():
    """Creates a parser for STL over atomic expressions of the type x_i ~ pi.

    Note that it is not restricted to depth-2 formulas.
    """
    stl_parser = MatchFirst(stl.stl_parser(expr_parser()))
    return stl_parser
