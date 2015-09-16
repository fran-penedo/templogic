from itertools import groupby
from stl import Signal, Formula, LE, GT, ALWAYS, EVENTUALLY, EXPR, AND, OR, NOT
from scipy import optimize


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
def lltinf(signals, robustness=None, depth=1):
    # Stopping condition
    if stop_inference(signals, depth):
        return None

    # Find primitive using impurity measure
    primitives = make_llt_primitives(signals)
    primitive, impurity = find_best_primitive(signals, primitives, robustness)

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
    return length(signals) == 0

def depth_stop(signals, depth):
    return depth <= 0

class LLTSignal(Signal):

    """Definition of a signal in LLT: x_j ~ pi, where ~ is <= or >"""

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

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value


class SimpleModel(object):

    """Matrix-like model"""

    def __init__(self, signals):
        self._signals = signals

    def getVarByName(self, indices):
        return self._signals[indices[0]][indices[1]]


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

def find_best_primitive(signals, primitives, robustness):
    # Parameters will be set for the copy of the primitive
    opt_prims = [optimize_impurity(signals, primitive.copy(), robustness)
                 for primitive in primitives]
    return max(opt_prims, key=lambda x: x[1])

optimize_purity = optimize_inf_gain

def optimize_inf_gain(signals, primitive, robustness):
    # [t0, t1, t3, pi]
    maxt = max(signals[-1])
    theta0 = [0, 0, 0, 0]
    lower = [0, 0, 0, min(signals[primitive.index])]
    upper = [maxt, maxt, maxt, min(signals[primitive.index])]
    args = [primitive, signals, robustness]

    res = optimize.anneal(inf_gain, theta0, args=args, lower=lower, upper=upper)
    return primitive, res[1]

def inf_gain(theta, *args):
    primitive = args[0]
    signals = args[1]
    # May be None, TODO check. Can't do it up in the stack
    robustness = args[2]

    set_llt_pars(primitive, theta[0], theta[1], theta[2], theta[3])

    # TODO use actual function
    return - theta[0] - theta[1] - theta[2] - theta[3] - primitive.index



