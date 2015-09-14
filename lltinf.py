from itertools import groupby

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
    # Find primitive using impurity measure
    primitive = find_best_primitive(signals)

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
