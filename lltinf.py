from itertools import groupby
from stl import Formula, AND, OR, NOT
from impurity import optimize_inf_gain
from llt import make_llt_primitives


class DTree(object):

    """Decission tree recursive structure"""

    def __init__(self, primitive, signals, robustness=None,
                 left=None, right=None):
        self._primitive = primitive
        self._signals = signals
        self._robustness = robustness
        self._left = left
        self._right = right

    def classify(self, signal):
        if satisfies(self.primitive, SimpleModel(signal)):
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

    @property
    def robustness(self):
        return self._robustness

    @robustness.setter
    def robustness(self, value):
        self._robustness = value


# Main inference function
def lltinf(signals, robustness=None, depth=1,
           optimize_impurity=optimize_inf_gain):
    # Stopping condition
    if stop_inference(signals, depth):
        return None

    # Find primitive using impurity measure
    primitives = make_llt_primitives(signals)
    primitive, impurity = find_best_primitive(signals, primitives, robustness,
                                              optimize_impurity)

    # Classify using best primitive
    # TODO add robustness
    tree = DTree(primitive, signals)
    classified = [(tree.classify(s[0]), s) for s in signals]

    # Split into groups
    grouped = dict(groupby(sorted(classified), lambda x: x[0]))
    sat = zip(*list(grouped[True]))[1]
    unsat = zip(*list(grouped[False]))[1]

    # Recursively build the tree
    tree.left = lltinf(sat, depth - 1)
    tree.right = lltinf(unsat, depth - 1)

    return tree


def stop_inference(signals, depth):
    stopping_conditions = [
        perfect_stop
    ]

    return any([stop(signals, depth) for stop in stopping_conditions])

def perfect_stop(signals, depth):
    return len(signals) == 0

def depth_stop(signals, depth):
    return depth <= 0

def find_best_primitive(signals, primitives, robustness):
    # Parameters will be set for the copy of the primitive
    opt_prims = [optimize_impurity(signals, primitive.copy(), robustness)
                 for primitive in primitives]
    return max(opt_prims, key=lambda x: x[1])

