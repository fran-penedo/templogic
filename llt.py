from stl import Signal, Formula, LE, GT, ALWAYS, EVENTUALLY, EXPR
import itertools
# from bisect import bisect_left
import numpy as np


class LLTSignal(Signal):

    """Definition of a signal in LLT: x_j ~ pi, where ~ is <= or >"""

    def __init__(self, index=0, op=LE, pi=0):
        self._index = index
        self._op = op
        self._pi = pi

        self._labels = [lambda t: [self._index, t]]
        self._f = lambda vs: (vs[0] - self._pi) * (-1 if self._op == LE else 1)

    def __deepcopy__(self, memo):
        return LLTSignal(self.index, self.op, self.pi)

    def __str__(self):
        return "x_%d %s %.2f" % (self.index,
                                 "<=" if self.op == LE else ">", self.pi)

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, value):
        self._pi = value

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, value):
        self._op = value

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value


class LLTFormula(Formula):

    """Docstring for LLTFormula. """

    def __init__(self, live, index, op):
        if live:
            Formula.__init__(self, ALWAYS, [
                Formula(EVENTUALLY, [
                    Formula(EXPR, [
                        LLTSignal(index, op)
                    ])
                ])
            ])
        else:
            Formula.__init__(self, EVENTUALLY, [
                Formula(ALWAYS, [
                    Formula(EXPR, [
                        LLTSignal(index, op)
                    ])
                ])
            ])

    @property
    def index(self):
        return self.args[0].args[0].args[0].index

    @property
    def pi(self):
        return self.args[0].args[0].args[0].pi

    @pi.setter
    def pi(self, value):
        self.args[0].args[0].args[0].pi = value

    @property
    def t0(self):
        return self.bounds[0]

    @t0.setter
    def t0(self, value):
        self.bounds[0] = value

    @property
    def t1(self):
        return self.bounds[1]

    @t1.setter
    def t1(self, value):
        self.bounds[1] = value

    @property
    def t3(self):
        return self.args[0].bounds[1]

    @t3.setter
    def t3(self, value):
        self.args[0].bounds[1] = value

    def reverse_op(self):
        op = self.args[0].args[0].args[0].op
        self.args[0].args[0].args[0].op = LE if op == GT else GT


def set_llt_pars(primitive, t0, t1, t3, pi):
    primitive.t0 = t0
    primitive.t1 = t1
    primitive.t3 = t3
    primitive.pi = pi


class SimpleModel(object):

    """Matrix-like model"""

    def __init__(self, signals):
        self._signals = signals
        self._tinter = signals[-1][1] - signals[-1][0]
        self._lsignals = len(signals[-1])

    def getVarByName(self, indices):
        '''indices[0] represents the name of the signal
        indices[1] represents the time at which to sample the signal
        '''
#         tindex = max(min(
#             bisect_left(self._signals[-1], indices[1]),
#             len(self._signals[-1]) - 1),
#             0)
        '''FIXME: Assumes that sampling rate is constant, i.e. the sampling
        times are in arithmetic progression with rate self._tinter'''
        tindex = min(
            np.floor(indices[1]/self._tinter), self._lsignals - 1)
        # assert 0 <= tindex <= len(self._signals[-1]), \
        #        'Invalid query outside the time domain of the trace! %f' % tindex
        return self._signals[indices[0]][tindex]

    @property
    def tinter(self):
        return self._tinter


def make_llt_primitives(signals):
    alw_ev = [
        LLTFormula(True, index, op)
        for index, op
        in itertools.product(range(len(signals[0]) - 1), [LE])
    ]
    ev_alw = [
        LLTFormula(False, index, op)
        for index, op
        in itertools.product(range(len(signals[0]) - 1), [LE])
    ]

    return alw_ev + ev_alw

def split_groups(l, group):
    p = [x for x in l if group(x)]
    n = [x for x in l if not group(x)]
    return p, n
