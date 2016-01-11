"""
Main inference module.

Contains the decision tree construction and related definitions.

Author: Francisco Penedo (franp@bu.edu)

"""
from stl import Formula, AND, OR, NOT, satisfies, robustness
from impurity import optimize_inf_gain
from llt import make_llt_primitives, split_groups, SimpleModel
import numpy as np

class Traces(object):
    """
    Class to store a set of labeled signals
    """

    def __init__(self, signals=None, labels=None):
        """
        signals : list of m by n matrices
                  Last row should be the sampling times
        labels : list of labels
                 Each label should be either 1 or -1
        """
        self._signals = [] if signals is None else np.array(signals, dtype=float)
        self._labels = [] if labels is None else labels

    @property
    def labels(self):
        return self._labels

    @property
    def signals(self):
        return self._signals

    def get_sindex(self, i):
        """
        Obtains the ith component of each signal

        i : integer
        """
        return self.signals[:, i]

    def as_list(self):
        """
        Returns the constructor arguments
        """
        return [self.signals, self.labels]


class DTree(object):
    """
    Decission tree recursive structure

    """

    def __init__(self, primitive, traces, robustness=None,
                 left=None, right=None):
        """
        primitive : a LLTFormula object
                    The node's primitive
        traces : a Traces object
                 The traces used to build this node
        robustness : a list of numeric. Not used
        left : a DTree object. Optional
               The subtree corresponding to an unsat result to this node's test
        right : a DTree object. Optional
                The subtree corresponding to a sat result to this node's test
        """
        self._primitive = primitive
        self._traces = traces
        self._robustness = robustness
        self._left = left
        self._right = right

    def classify(self, signal):
        """
        Classifies a signal. Returns a label 1 or -1

        signal : an m by n matrix
                 Last row should be the sampling times
        """
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
        """
        Obtains an STL formula equivalent to this tree
        """
        left = self.primitive
        right = Formula(NOT, [self.primitive])
        if self.left is not None:
            left = Formula(AND, [
                self.primitive,
                self.left.get_formula()
            ])
        if self.right is not None:
            return Formula(OR, [left,
                                Formula(AND, [
                                    right,
                                    self.right.get_formula()
            ])])
        else:
            return left

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
def lltinf(traces, depth=1,
           optimize_impurity=optimize_inf_gain, stop_condition=None, disp=False):
    """
    Obtains a decision tree that classifies the given labeled traces.

    traces : a Traces object
             The set of labeled traces to use as training set
    depth : integer
            Maximum depth to be reached
    optimize_impurity : function. Optional, defaults to optimize_inf_gain
                        A function that obtains the best parameters for a test
                        in a given node according to some impurity measure. The
                        should have the following prototype:
                            optimize_impurity(traces, primitive, rho, disp) :
                                (primitive, impurity)
                        where traces is a Traces object, primitive is a depth 2
                        STL formula, rho is a list with the robustness degree of
                        each trace up until this node in the tree and disp is a
                        boolean that switches output display. The impurity
                        returned should be so that the best impurity is the
                        minimum one.
    stop_condition : list of functions. Optional, defaults to [perfect_stop]
                     list of stopping conditions. Each stopping condition is a
                     function from a dictionary to boolean. The dictionary
                     contains all the information passed recursively during the
                     construction of the decision tree (see arguments of
                     lltinf_).
    disp : a boolean
           Switches displaying of debuggin output

    Returns a DTree object.

    TODO: This should also be parameterized by make_llt_primitives

    """
    np.seterr(all='ignore')
    if stop_condition is None:
        stop_condition = [perfect_stop]

    return lltinf_(traces, None, depth, optimize_impurity, stop_condition, disp)

def lltinf_(traces, rho, depth, optimize_impurity, stop_condition, disp=False):
    """
    Recursive call for the decision tree construction.

    See lltinf for information on similar arguments.

    rho : list of numerics
          List of robustness values for each trace up until the current node
    depth : integer
            Maximum depth to be reached. Decrements for each recursive call
    """
    args = locals().copy()
    # Stopping condition
    if any([stop(args) for stop in stop_condition]):
        return None

    # Find primitive using impurity measure
    primitives = make_llt_primitives(traces.signals)
    primitive, impurity = _find_best_primitive(traces, primitives, rho,
                                              optimize_impurity, disp)
    if disp:
        print primitive

    # Classify using best primitive and split into groups
    tree = DTree(primitive, traces)
    prim_rho = [robustness(primitive, SimpleModel(s)) for s in traces.signals]
    if rho is None:
        rho = [np.inf for i in traces.labels]
    # [prim_rho, rho, signals, label]
    sat_, unsat_ = split_groups(zip(prim_rho, rho, *traces.as_list()),
        lambda x: x[0] >= 0)

    # Switch sat and unsat if labels are wrong. No need to negate prim rho since
    # we use it in absolute value later
    if len([t for t in sat_ if t[3] >= 0]) < \
        len([t for t in unsat_ if t[3] >= 0]):
        sat_, unsat_ = unsat_, sat_
        tree.primitive = Formula(NOT, [tree.primitive])

    # No further classification possible
    if len(sat_) == 0 or len(unsat_) == 0:
        return None

    # Redo data structures
    sat, unsat = [(Traces(*group[2:]),
                   np.amin([np.abs(group[0]), group[1]], 0))
                   for group in [zip(*sat_), zip(*unsat_)]]

    # Recursively build the tree
    tree.left = lltinf_(sat[0], sat[1], depth - 1,
                        optimize_impurity, stop_condition)
    tree.right = lltinf_(unsat[0], unsat[1], depth - 1,
                         optimize_impurity, stop_condition)

    return tree

def perfect_stop(kwargs):
    """
    Returns True if all traces are equally labeled.
    """
    return all([l > 0 for l in kwargs['traces'].labels]) or \
        all([l <= 0 for l in kwargs['traces'].labels])

def depth_stop(kwargs):
    """
    Returns True if the maximum depth has been reached
    """
    return kwargs['depth'] <= 0

def _find_best_primitive(traces, primitives, robustness, optimize_impurity, disp):
    # Parameters will be set for the copy of the primitive
    opt_prims = [optimize_impurity(traces, primitive.copy(), robustness, disp)
                 for primitive in primitives]
    return min(opt_prims, key=lambda x: x[1])

