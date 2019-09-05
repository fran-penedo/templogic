from typing import Tuple, Sequence, Optional, Union, TypeVar
import copy

import numpy as np  # type: ignore

from templogic import tssl, stlmilp as stl
from templogic.util import round_t
from . import spatel

SignalType = Sequence[Tuple[float, tssl.QuadTree]]


class SpatelTraces(stl.inference.Traces):
    def __init__(
        self, signals: Sequence[SignalType], labels: Sequence[int] = None
    ) -> None:
        super().__init__(signals, labels)

    def model(
        self, signal: SignalType, interpolate: bool, tinter: float
    ) -> stl.STLModel:
        # FIXME add support for interpolation
        return spatel.SpatelModel([tree for t, tree in signal], self.data_abs_bounds)

    @property
    def data_abs_bounds(self) -> Tuple[float, ...]:
        return tuple([max(abs(a), abs(b)) for a, b in self.data_bounds])

    def add_traces(self, signals: Sequence[SignalType], labels: Sequence[int]) -> None:
        if len(self) == 0:
            self._signals = list(signals)
            self._labels = np.array(labels)
        else:
            self._signals.extend(signals)
            self._labels = np.hstack([self._labels, np.array(labels)])
        # mins, maxs = np.amin(signals, (2, 0)), np.amax(signals, (2, 0))
        # mint, maxt = self.time_bounds
        # self._time_bounds = (min(mint, mins[-1]), max(maxt, maxs[-1]))
        # if len(self._data_bounds) == 0:
        #     self._data_bounds = [(np.inf, -np.inf) for i in range(len(mins) - 1)]
        # self._data_bounds = [
        #     (min(cur[0], mins[i]), max(cur[1], maxs[i]))
        #     for i, cur in enumerate(self._data_bounds)
        # ]
        mint, maxt = self.time_bounds
        ts = [t for signal in signals for t in list(zip(*signal))[0]]
        self._time_bounds = (min(mint, min(ts)), max(maxt, max(ts)))


T = TypeVar("T")


class SpatelAbstractPrimitive(  # type: ignore # copy issues
    spatel.SpatelTerm, stl.inference.Primitive[T]
):
    def copy(self) -> "SpatelAbstractPrimitive":
        return copy.deepcopy(self)

    def score(self, model: stl.STLModel) -> float:
        if not isinstance(model, spatel.SpatelModel):
            raise AttributeError("Model should be a spatel model")
        return spatel.robustness(self, model)


class SpatelPrimitiveD1(  # type: ignore # copy issues
    spatel.STLAnd, SpatelAbstractPrimitive[Tuple[float, float]]
):
    outer: Union[spatel.STLAlways, spatel.STLEventually]
    inner: spatel.SpatelSTLPred

    def __init__(self, safe: bool, pred: spatel.SpatelSTLPred):
        self.inner = pred
        if safe:
            self.outer = spatel.STLAlways((0, 0), pred)
        else:
            self.outer = spatel.STLEventually((0, 0), pred)
        super().__init__([self.outer])

    def parameter_bounds(
        self,
        time_bounds: Tuple[float, float],
        data_bounds: Sequence[Tuple[float, float]],
    ) -> Sequence[Tuple[float, float]]:
        return [time_bounds]

    def set_pars(
        self,
        parameters: Tuple[float, float],
        time_bounds: Tuple[float, float],
        times: Optional[Sequence[float]],
    ) -> None:
        t0, t1 = parameters
        maxt = time_bounds[1]
        if maxt > 0:
            t1 = t0 + (maxt - t0) * t1 / maxt
            t0, t1 = [round_t(t, times) for t in [t0, t1]]
        else:
            t0 = t1 = 0.0
        self.outer.bounds = (t0, t1)

    def negate(self) -> None:
        self.inner.negate()


def make_tssl_primitives(signals):
    pass


class SpatelInference(stl.inference.inference.LLTInf):
    def __init__(self):
        super().__init__(primitive_factory=make_tssl_primitives)
