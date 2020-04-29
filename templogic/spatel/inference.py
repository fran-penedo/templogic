from typing import Tuple, Sequence, Optional, Union, TypeVar, Iterator, Type
import copy
import itertools as it

import numpy as np
import attr

from templogic import tssl, stlmilp as stl
from templogic.util import round_t
from . import spatel

SignalType = Sequence[Tuple[float, tssl.QuadTree]]
MatrixSignalType = Sequence[Tuple[float, np.ndarray]]


class SpatelTraces(stl.inference.Traces):
    def __init__(
        self, signals: Sequence[SignalType], labels: Sequence[int] = None
    ) -> None:
        if len(signals) > 0 and not isinstance(signals[0][0][1], tssl.QuadTree):
            raise TypeError("Signal value should be a QuadTree")
        super().__init__(signals, labels)

    @classmethod
    def from_matrices(
        cls: Type["SpatelTraces"],
        signals: Sequence[MatrixSignalType],
        labels: Sequence[int] = None,
    ) -> "SpatelTraces":
        signals = [[(t, tssl.QuadTree.from_matrix(m)) for t, m in s] for s in signals]
        return cls(signals, labels)

    def model(
        self, signal: SignalType, interpolate: bool, tinter: float
    ) -> stl.STLModel:
        # FIXME add support for interpolation
        return spatel.SpatelModel([tree for t, tree in signal], self.data_abs_bounds)

    @property
    def data_abs_bounds(self) -> Tuple[float, ...]:
        return tuple([max(abs(a), abs(b)) for a, b in self.data_bounds])

    def add_traces(self, signals: Sequence[SignalType], labels: Sequence[int]) -> None:
        if len(signals) > 0 and not isinstance(signals[0][0][1], tssl.QuadTree):
            raise TypeError("Signal value should be a QuadTree")
        if len(self) == 0:
            self._signals = list(signals)
            self._labels = np.array(labels)
        else:
            self._signals.extend(signals)
            self._labels = np.hstack([self._labels, np.array(labels)])
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
        return [time_bounds, time_bounds]

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


@attr.s(auto_attribs=True)
class SplitData(object):
    data: Sequence[SignalType]
    labels: Sequence[int]
    # Length of a consecutive set of images from a signal
    chunk_len: int = 4
    # Number of chunks to get for each signal
    nchunks: int = 1
    # Divide data length by this to get the number of signals for each dataset
    nsignals_groups: int = 20
    # Number of datasets to generate
    ndatasets: int = 4

    def __attrs_post_init__(self) -> None:
        sorted_list = sorted(zip(self.data, self.labels), key=lambda x: x[1])
        self.data, self.labels = list(zip(*sorted_list))
        self.data_len = len(self.labels)
        self.nsignals = self.data_len // self.nsignals_groups

        self.valid_set, self.invalid_set = [
            list(zip(*group))[0]
            for k, group in it.groupby(zip(self.data, self.labels), key=lambda x: x[1])
        ]
        self.iteration = 0

    def __iter__(self):
        return self

    def _get_random_imgs(self, signals: Sequence[SignalType]) -> Iterator[np.ndarray]:
        for signal in signals:
            for j in range(self.nchunks):
                p = np.random.randint(0, len(signal) - self.chunk_len)
                for t, qt in signal[p : p + self.chunk_len]:
                    yield qt.to_matrix()

    def __next__(self) -> tssl.inference.Instances:
        if self.iteration >= self.ndatasets:
            raise StopIteration()
        self.iteration += 1
        valid_perm = np.random.permutation(len(self.valid_set))[: self.nsignals // 2]
        invalid_perm = np.random.permutation(len(self.invalid_set))[
            : self.nsignals // 2
        ]
        signals = [
            signal_set[j]
            for signal_set, perm in zip(
                [self.valid_set, self.invalid_set], [valid_perm, invalid_perm]
            )
            for j in perm
        ]
        imgs = np.array(list(self._get_random_imgs(signals)))
        nimgs_signal = self.nchunks * self.chunk_len
        nimgs_half = self.nsignals * nimgs_signal // 2
        labels = ["1"] * nimgs_half + ["0"] * nimgs_half

        dataset = tssl.inference.build_dataset(imgs, labels)
        return dataset


def make_tssl_primitives(
    signals: Sequence[SignalType], labels: Sequence[int]
) -> Sequence[SpatelAbstractPrimitive]:
    # TODO possibly add default (parametric?) primitives
    prims = []
    # FIXME GC issues
    tsslclassifier = tssl.inference.TSSLInference()
    for data in SplitData(signals, labels):
        tsslclassifier.build_classifier(data, "1")
        pred = spatel.SpatelSTLPred(tsslclassifier.get_tssl_formula())
        for safe in (True, False):
            prims.append(SpatelPrimitiveD1(safe, pred))

    return prims


class SpatelInference(stl.inference.inference.LLTInf):
    def __init__(self, **kwargs):
        kwargs.setdefault("primitive_factory", make_tssl_primitives)
        if "optimizer_args" in kwargs:
            kwargs["optimizer_args"].setdefault("workers", 1)
        else:
            kwargs["optimizer_args"] = {"workers": 1}
        super().__init__(**kwargs)
        self._debug("Created SpaTeL classifier")
