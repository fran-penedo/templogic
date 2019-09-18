from typing import Tuple, Sequence, Optional, Union, TypeVar, Iterator
import copy
import itertools as it

import numpy as np
import attr

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


@attr.s(auto_attribs=True, slots=True)
class SplitData(object):
    data: Sequence[SignalType]
    labels: Sequence[int]

    def __attrs_post_init__(self) -> None:
        sorted_list = sorted(zip(self.data, self.labels), key=lambda x: x[1])
        self.data, self.labels = list(zip(*sorted_list))

    def __iter__(self):
        return self

    def __next__(self) -> Iterator[tssl.inference.Instances]:
        data_len = len(self.labels)
        # Length of a consecutive set of images from a signal
        chunk_len = 4
        # Number of chunks to get for each signal
        nchunks = 4
        # Number of signals to get chunks from for each dataset
        nsignals = data_len // 20
        # Number of datasets to generate
        ndatasets = 10

        def _get_random_imgs(signals: Sequence[SignalType]) -> Iterator[np.ndarray]:
            for signal in signals:
                for j in range(nchunks):
                    p = np.randint(0, len(signal) - chunk_len)
                    for t, qt in signal[p : p + chunk_len]:
                        yield qt.flatten()

        valid_set, invalid_set = [
            list(zip(*group))[0] for k, group in it.groupby(zip(self.data, self.labels))
        ]
        for i in range(ndatasets):
            data_perm = np.random.permutation(data_len)[:nsignals]
            mid = len(data_perm) // 2
            signals = [
                signal_set[j]
                for signal_set, perm in zip(
                    [valid_set, invalid_set], [data_perm[:mid], data_perm[mid:]]
                )
                for j in perm
            ]
            imgs = np.array(_get_random_imgs(signals))
            labels = ["1"] * mid * nchunks + ["0"] * (data_len - mid) * nchunks

            dataset = tssl.inference.build_dataset(imgs, labels)
            yield dataset


def make_tssl_primitives(
    signals: Sequence[SignalType], labels: Sequence[int]
) -> Sequence[SpatelAbstractPrimitive]:
    tsslclassifier = tssl.inference.TSSLInference()
    # TODO possibly add default (parametric?) primitives
    prims = []
    for data in SplitData(signals, labels):
        tsslclassifier.build_classifier(data, "1")
        pred = spatel.SpatelSTLPred(tsslclassifier.get_tssl_formula())
        for safe in (True, False):
            prims.append(SpatelPrimitiveD1(safe, pred))

    return prims


class SpatelInference(stl.inference.inference.LLTInf):
    def __init__(self):
        super().__init__(primitive_factory=make_tssl_primitives)
