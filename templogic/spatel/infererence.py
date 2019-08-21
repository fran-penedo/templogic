from typing import Tuple

from templogic import tssl, stlmilp as stl


class SpatelTraces(Traces):
    pass


class TSSLPrimitive(TSSLTerm, Primitive):
    def copy(self):
        pass

    def parameter_bounds(self, traces_maybe):
        pass

    def set_pars(self, parameters):
        pass


def make_tssl_primitives(signals):
    pass


class SpatelInference(stl.inference.inference.LLTInf):
    def __init__(self):
        super().__init__(primitive_factory=make_tssl_primitives)
