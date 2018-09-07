"""
Module with depth 2 p-stl definitions

Author: Francisco Penedo (franp@bu.edu)

"""
import stlmilp.stl as stl
from stlmilp.stl import Signal, Formula, LE, GT, ALWAYS, EVENTUALLY, EXPR
import itertools
# from bisect import bisect_left
import numpy as np
from pyparsing import Word, alphas, Suppress, Optional, Combine, nums, \
    Literal, alphanums, Keyword, Group, ParseFatalException, MatchFirst


class LLTSignal(Signal):
    """
    Definition of an atomic proposition in LLT: x_j ~ pi, where ~ is <= or >
    """

    def __init__(self, index=0, op=LE, pi=0):
        """
        Creates a signal x_j ~ pi.

        index : integer
                Corresponds to j.
        op : either LE or GT
             Corresponds to op.
        pi : numeric

        """
        self._index = index
        self._op = op
        self._pi = pi

        # labels transform a time into a pair [j, t]
        self.labels = [lambda t: [self._index, t]]
        # transform to x_j - pi >= 0
        self.f = lambda vs: (vs[0] - self._pi) * (-1 if self._op == LE else 1)

    def __deepcopy__(self, memo):
        return LLTSignal(self.index, self.op, self.pi)

    def __str__(self):
        return "x_%d %s %f" % (self.index,
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
    """
    A depth 2 STL formula.
    """

    def __init__(self, live, index, op):
        """
        Creates a depth 2 STL formula.

        live : boolean
               True if the formula has a liveness structure (always eventually)
               or not (eventually always).
        index : integer
                Index for the signal (see LLTSignal)
        op : either LE or GT
             Operator for the atomic proposition (see LLTSignal)
        """
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
        """
        Reverses the operator of the atomic proposition
        """
        op = self.args[0].args[0].args[0].op
        self.args[0].args[0].args[0].op = LE if op == GT else GT


def set_llt_pars(primitive, t0, t1, t3, pi):
    """
    Sets the parameters of a primitive
    """
    primitive.t0 = t0
    primitive.t1 = t1
    primitive.t3 = t3
    primitive.pi = pi


class SimpleModel(object):
    """
    Matrix-like model with fixed sample interval. Suitable for use with
    LLTFormula
    """

    def __init__(self, signals):
        """
        signals : m by n matrix.
                  Last row should be the sampling times.
        """
        self._signals = signals
        self._lsignals = len(signals[-1])
        if self._lsignals == 1:
            self._tinter = 1.0
        else:
            self._tinter = signals[-1][1] - signals[-1][0]

    def getVarByName(self, indices):
        """
        indices : pair of numerics
                  indices[0] represents the name of the signal
                  indices[1] represents the time at which to sample the signal
        """
#         tindex = max(min(
#             bisect_left(self._signals[-1], indices[1]),
#             len(self._signals[-1]) - 1),
#             0)
        '''FIXME: Assumes that sampling rate is constant, i.e. the sampling
        times are in arithmetic progression with rate self._tinter'''
        if self._lsignals == 1:
            tindex = 0
        else:
            tindex = int(min(
                np.floor(indices[1]/self._tinter), self._lsignals - 1))
        # assert 0 <= tindex <= len(self._signals[-1]), \
        #        'Invalid query outside the time domain of the trace! %f' % tindex
        return self._signals[indices[0]][tindex]

    @property
    def tinter(self):
        return self._tinter


def make_llt_primitives(signals):
    """
    Obtains the depth 2 primitives associated with the structure of the signals.

    signals : m by n matrix
              Last column should be the sampling times
    """
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
    """
    Splits a list according to a binary grouping function. Returns the positive
    group first

    l : a list
    group : a function from elements of l to boolean
    """
    p = [x for x in l if group(x)]
    n = [x for x in l if not group(x)]
    return p, n


# parser

def expr_parser():
    num = stl.num_parser()

    T_UND = Suppress(Literal("_"))
    T_LE = Literal("<=")
    T_GR = Literal(">")

    integer = Word(nums).setParseAction(lambda t: int(t[0]))
    relation = (T_LE | T_GR).setParseAction(lambda t: LE if t[0] == "<=" else GT)
    expr = Suppress(Word(alphas)) + T_UND + integer + relation + num
    expr.setParseAction(lambda t: LLTSignal(t[0], t[1], t[2]))

    return expr

def llt_parser():
    """
    Creates a parser for STL over atomic expressions of the type x_i ~ pi.

    Note that it is not restricted to depth-2 formulas.
    """
    stl_parser = MatchFirst(stl.stl_parser(expr_parser()))
    return stl_parser
