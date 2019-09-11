"""Module with depth 2 p-stl definitions

Author: Francisco Penedo (franp@bu.edu)

"""
import logging
from abc import ABC, abstractmethod
from bisect import bisect_right
import itertools
from enum import Enum
import operator
from typing import Iterable, Sequence, Tuple, Union, cast, TypeVar, Generic, Optional

import numpy as np
from pyparsing import Word, alphas, Suppress, nums, Literal, MatchFirst  # type: ignore

from .. import stl
from templogic.util import round_t

logger = logging.getLogger(__name__)


T = TypeVar("T")


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


class Primitive(ABC, Generic[T]):
    @abstractmethod
    def copy(self) -> "Primitive":
        pass

    @abstractmethod
    def parameter_bounds(
        self,
        time_bounds: Tuple[float, float],
        data_bounds: Sequence[Tuple[float, float]],
    ) -> Sequence[Tuple[float, float]]:
        pass

    @abstractmethod
    def set_pars(
        self,
        parameters: T,
        time_bounds: Tuple[float, float],
        times: Optional[Sequence[float]],
    ) -> None:
        pass

    @abstractmethod
    def negate(self) -> None:
        pass

    @abstractmethod
    def score(self, model: stl.STLModel) -> float:
        pass

    def satisfies(self, model: stl.STLModel) -> bool:
        return self.score(model) >= 0


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
        super().__init__(self._labels, self._f)
        self.index = index
        self.op = op
        self.pi = pi

    def _labels(self, t):
        # labels transform a time into a pair [j, t]
        return [[self._index, t]]

    def _f(self, vs):
        # transform to x_j - pi >= 0
        return (vs[0] - self._pi) * (-1 if self.op == Relation.LE else 1)

    def __deepcopy__(self, memo):
        return LLTSignal(self.index, self.op, self.pi)

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


class LLTFormulaD1(  # type: ignore # copy issues
    stl.STLAnd, Primitive[Tuple[float, float, float]]
):
    signal: LLTSignal
    outer: stl.TemporalTerm

    def __init__(self, safe: bool, index: int, op: Union[str, Relation]) -> None:
        """ Creates a depth 1 STL formula.

        safe : boolean
               True if the formula has a safety structure (always)
               or not (eventually).
        index : integer
                Index for the signal (see LLTSignal)
        op : either LE or GT
             Operator for the atomic proposition (see LLTSignal)
        """
        self.signal = LLTSignal(index, op)
        if safe:
            self.outer = stl.STLAlways((0, 0), stl.STLPred(self.signal))
        else:
            self.outer = stl.STLEventually((0, 0), stl.STLPred(self.signal))
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

    def negate(self) -> None:
        """Reverses the operator of the atomic proposition
        """
        self.op = self.op.flip()

    def copy(self) -> "LLTFormulaD1":
        return cast("LLTFormulaD1", super().copy())

    def parameter_bounds(
        self,
        time_bounds: Tuple[float, float],
        data_bounds: Sequence[Tuple[float, float]],
    ) -> Sequence[Tuple[float, float]]:
        return [time_bounds, time_bounds, data_bounds[self.index]]

    def set_pars(
        self,
        parameters: Tuple[float, float, float],
        time_bounds: Tuple[float, float],
        times: Optional[Sequence[float]],
    ) -> None:
        t0, t1, pi = parameters
        maxt = time_bounds[1]
        if maxt > 0:
            t1 = t0 + (maxt - t0) * t1 / maxt
            t0, t1 = [round_t(t, times) for t in [t0, t1]]
        else:
            t0 = t1 = 0.0

        self.set_llt_pars((t0, t1, pi))

    def set_llt_pars(self, theta: Tuple[float, float, float]) -> None:
        """ Sets the parameters of a primitive
        """
        t0, t1, pi = theta
        self.t0 = t0
        self.t1 = t1
        self.pi = pi

    def score(self, model: stl.STLModel) -> float:
        return stl.robustness(self, model)


class LLTFormula(  # type: ignore # copy issues
    stl.STLAnd, Primitive[Tuple[float, float, float, float]]
):
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

    def negate(self) -> None:
        """Reverses the operator of the atomic proposition
        """
        self.op = self.op.flip()

    def copy(self) -> "LLTFormula":
        return cast("LLTFormula", super().copy())

    def parameter_bounds(
        self,
        time_bounds: Tuple[float, float],
        data_bounds: Sequence[Tuple[float, float]],
    ) -> Sequence[Tuple[float, float]]:
        return [time_bounds, time_bounds, time_bounds, data_bounds[self.index]]

    def set_pars(
        self,
        parameters: Tuple[float, float, float, float],
        time_bounds: Tuple[float, float],
        times: Optional[Sequence[float]],
    ) -> None:
        t0, t1, t3, pi = parameters
        maxt = time_bounds[1]
        if maxt > 0:
            t1 = t0 + (maxt - t0) * t1 / maxt
            t3 = (maxt - t1) * t3 / maxt
            t0, t1, t3 = [round_t(t, times) for t in [t0, t1, t3]]
        else:
            t0 = t1 = t3 = 0.0
        self.set_llt_pars((t0, t1, t3, pi))

    def set_llt_pars(self, theta: Tuple[float, float, float, float]) -> None:
        """Sets the parameters of a primitive
        """
        t0, t1, t3, pi = theta
        self.t0 = t0
        self.t1 = t1
        self.t3 = t3
        self.pi = pi

    def score(self, model: stl.STLModel):
        return stl.robustness(self, model)


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
    interpolate: bool

    def __init__(self, signal: SignalType, interpolate=False, tinter=None) -> None:
        self._signal = signal
        try:
            _tinter = signal[-1][1] - signal[-1][0]
        except IndexError:  # only one time
            _tinter = 1
        self.tinter = tinter or _tinter
        self.interpolate = interpolate
        self._lsignal = len(signal[-1])

    def getVarByName(self, indices: Tuple[int, float]) -> float:
        """ Get variables

        indices : pair of numerics
                  indices[0] represents the name of the signal
                  indices[1] represents the time at which to sample the signal

        FIXME: Assumes that sampling rate is constant, i.e. the sampling
        times are in arithmetic progression with rate self._tinter
        """
        name, time = indices
        #         tindex = max(min(
        #             bisect_left(self._signals[-1], indices[1]),
        #             len(self._signals[-1]) - 1),
        #             0)
        if self._lsignal == 1:
            return self._signal[name][0]
        elif self.interpolate:
            times = self._signal[-1]
            signal = self._signal[name]
            if time == times[0]:
                return signal[0]
            elif time == times[-1]:
                return signal[-1]
            else:
                tindex = bisect_right(times, time) - 1
                tinter = times[tindex + 1] - times[tindex]
                lam = (time - times[tindex]) / tinter
                ret = (1 - lam) * signal[tindex] + lam * signal[tindex + 1]
                # logger.debug([name, time, tindex, tinter, ret])
                return ret
        else:
            tindex = int(min(np.floor(time / self.tinter), self._lsignal - 1))
            # assert 0 <= tindex <= len(self._signals[-1]), \
            #        'Invalid query outside the time domain of the trace! %f' % tindex
            return self._signal[name][tindex]


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


def make_llt_d1_primitives(signals: Sequence[SignalType]) -> Iterable[LLTFormulaD1]:
    """ Obtains the depth 1 primitives associated with the structure of the signals.

    signals : m by n matrix
              Last column should be the sampling times
    """
    prims = [
        LLTFormulaD1(safe, index, op)
        for safe in [True, False]
        for index, op in itertools.product(range(len(signals[0]) - 1), [Relation.LE])
    ]

    return prims


# parser


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
