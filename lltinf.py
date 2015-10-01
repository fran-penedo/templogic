from stl import Formula, AND, OR, NOT, satisfies, robustness
from impurity import optimize_inf_gain
from llt import make_llt_primitives, split_groups, SimpleModel
import numpy as np

class Traces(object):

    def __init__(self, signals=None, labels=None):
        self._signals = [] if signals is None else np.array(signals, dtype=float)
        self._labels = [] if labels is None else labels

    @property
    def labels(self):
        return self._labels

    @property
    def signals(self):
        return self._signals


    def get_sindex(self, i):
        return self.signals[:, i]

    def as_list(self):
        return [self.signals, self.labels]


class DTree(object):

    """Decission tree recursive structure"""

    def __init__(self, primitive, traces, robustness=None,
                 left=None, right=None):
        self._primitive = primitive
        self._traces = traces
        self._robustness = robustness
        self._left = left
        self._right = right

    def classify(self, signal):
        if satisfies(self.primitive, SimpleModel(signal)):
            if self.left is None:
                return 1
            else:
                return self.left.classify(signal)
        else:
            if self.right is None:
                return -1
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

    @property
    def robustness(self):
        return self._robustness

    @robustness.setter
    def robustness(self, value):
        self._robustness = value


# Main inference function
def lltinf(traces, rho=None, depth=1,
           optimize_impurity=optimize_inf_gain, stop_condition=None):
    np.seterr(all='ignore')
    if stop_condition is None:
        stop_condition = [perfect_stop]

    return lltinf_(traces, rho, depth, optimize_impurity, stop_condition)

def lltinf_(traces, rho, depth, optimize_impurity, stop_condition):
    args = locals().copy()
    # Stopping condition
    if any([stop(args) for stop in stop_condition]):
        return None

    # Find primitive using impurity measure
    primitives = make_llt_primitives(traces.signals)
    primitive, impurity = find_best_primitive(traces, primitives, rho,
                                              optimize_impurity)

    # Classify using best primitive and split into groups
    tree = DTree(primitive, traces)
    prim_rho = [robustness(primitive, SimpleModel(s)) for s in traces.signals]
    if rho is None:
        rho = [np.inf for i in traces.labels]
    sat_, unsat_ = split_groups(zip(prim_rho, rho, traces.as_list()),
        lambda x: x[0] >= 0)

    # No further classification possible
    if len(sat_) == 0 or len(unsat_) == 0:
        return None

    # Redo data structures
    sat, unsat = [(Traces(*group[2:]),
                   np.amin([np.abs(group[0]), group[1][:,primitive.index]], 1))
                   for group in [zip(*sat), zip(*unsat)]]

    # Recursively build the tree
    tree.left = lltinf_(sat, rho_sat, depth - 1, optimize_impurity)
    tree.right = lltinf_(unsat, rho_unsat, depth - 1, optimize_impurity)

    return tree

def perfect_stop(kwargs):
    return len(kwargs['traces'].signals) == 0

def depth_stop(kwargs):
    return kwargs['depth'] <= 0

def find_best_primitive(traces, primitives, robustness, optimize_impurity):
    # Parameters will be set for the copy of the primitive
    opt_prims = [optimize_impurity(traces, primitive.copy(), robustness)
                 for primitive in primitives]
    return max(opt_prims, key=lambda x: x[1])

