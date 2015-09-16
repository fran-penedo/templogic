from itertools import groupby
from stl import Signal, Formula, LE, GT, ALWAYS, EVENTUALLY, EXPR, AND, OR, NOT


class DTree(object):

    """Decission tree recursive structure"""

    def __init__(self, primitive, signals, left=None, right=None):
        self._primitive = primitive
        self._signals = signals
        self._left = left
        self._right = right

    def classify(self, signal):
        if self.primitive.sats(signal):
            return self.left.classify(signal)
        else:
            return self.right.classify(signal)

    def get_formula(self):
        return Formula(OR, [
            Formula(AND, [
                self.primitive,
                self.left.get_formula()
            ]),
            Formula(AND, [
                Formula(NOT, [self.primitive]),
                self.right
            ])
        ])

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value

    @property
    def primitive(self):
        return self._primitive

    @primitive.setter
    def primitive(self, value):
        self._primitive = value



def lltinf(signals):
    # Stopping condition
    if stop_inference(signals):
        return None

    # Find primitive using impurity measure
    primitives = make_llt_primitives(signals)
    primitive = find_best_primitive(signals, primitives)

    # Classify using best primitive
    tree = DTree(primitive, signals)
    classified = [(tree.classify(s[0]), s) for s in signals]

    # Split into groups
    grouped = dict(groupby(sorted(classified), lambda x: x[0]))
    sat = zip(*list(grouped[True]))[1]
    unsat = zip(*list(grouped[False]))[1]

    # Recursively build the tree
    tree.left = lltinf(sat)
    tree.right = lltinf(unsat)

    return tree


def stop_inference(signals):
    stopping_conditions = [
        perfect_stop
    ]

    return any([stop(signals) for stop in stopping_conditions])

def perfect_stop(signals):
    return length(signals) == 0

class SimpleSignal(Signal):

    def __init__(self, index=0, op=LE, pi=0):
        self._index = index
        self._op = op
        self._pi = pi

        self._labels = lambda t: [self._index, t]
        self._f = lambda vs: (vs[0] - self._pi) * (-1 if self._op == LE else 1)

    @property
    def pi(self):
        return self._pi

    @pi.setter
    def pi(self, value):
        self._pi = value

def make_llt_primitives(signals):
    alw_ev = [
        Formula(ALWAYS, [
            Formula(EVENTUALLY, [
                Formula(EXPR, [
                    SimpleSignal(index, op)
                ])
            ])
        ])
        for index, op
        in itertools.product(range(length(signals[0] - 1), [LE, GT]))
    ]
    ev_alw = [
        Formula(EVENTUALLY, [
            Formula(ALWAYS, [
                Formula(EXPR, [
                    SimpleSignal(index, op)
                ])
            ])
        ])
        for index, op
        in itertools.product(range(length(signals[0] - 1), [LE, GT]))
    ]

    return alw_ev + ev_alw

def set_llt_pars(primitive, t0, t1, t3, pi):
    primitive.bounds = [t0, t1]
    primitive.args[0].bounds = [0, t3]
    primitives.args[0].args[0].pi = pi

