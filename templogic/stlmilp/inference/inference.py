""" Main inference module.

Contains the decision tree construction and related definitions.

Author: Francisco Penedo (franp@bu.edu)

"""
import logging
from multiprocessing.pool import Pool
import signal

import numpy as np  # type: ignore

from .. import stl, llt
from . import impurity
from templogic.util import split_groups

logger = logging.getLogger(__name__)


class Traces(object):
    """ Class to store a set of labeled signals
    """

    def __init__(self, signals=None, labels=None):
        """ signals : list of m by n matrices
                  Last row should be the sampling times
        labels : list of labels
                 Each label should be either 1 or -1
        """
        self._signals = (
            np.array([], dtype=float)
            if signals is None
            else np.array(signals, dtype=float)
        )
        self._labels = np.array([] if labels is None else list(labels))

    @property
    def labels(self):
        return self._labels

    @property
    def signals(self):
        return self._signals

    def get_sindex(self, i):
        """ Obtains the ith component of each signal

        i : integer
        """
        return self.signals[:, i]

    def as_list(self):
        """ Returns the constructor arguments
        """
        return [self.signals, self.labels]

    def zipped(self):
        """ Returns the constructor arguments zipped
        """
        return zip(*self.as_list())

    def add_traces(self, signals, labels):
        self._signals = np.vstack([self._signals, signals])
        self._labels = np.hstack([self._labels, labels])

    def copy(self):
        return Traces(self._signals.copy(), self._labels.copy())

    def __len__(self):
        return len(self.signals)


class DTree(object):
    """ Decission tree recursive structure

    """

    def __init__(self, primitive, traces, robustness=None, left=None, right=None):
        """ primitive : a LLTFormula object
                    The node's primitive
        traces : a Traces object
                 The traces used to build this node
        robustness : a list of numeric. Not used
        left : a DTree object. Optional
               The subtree corresponding to an unsat result to this node's test
        right : a DTree object. Optional
                The subtree corresponding to a sat result to this node's test
        """
        self.primitive = primitive
        self.traces = traces
        self.robustness = np.array(robustness)
        self._left = left
        self._right = right
        self.parent = None

    def set_tree(self, tree):
        self.primitive = tree.primitive
        self.traces = tree.traces
        self.robustness = tree.robustness
        self.left = tree.left
        self.right = tree.right

    def copy(self):
        return DTree(
            self.primitive, self.traces, self.robustness, self.left, self.right
        )

    def deep_copy(self):
        left = None if self.left is None else self.left.deep_copy()
        right = None if self.right is None else self.right.deep_copy()
        return DTree(
            self.primitive.copy(),
            self.traces.copy(),
            self.robustness.copy(),
            left,
            right,
        )

    def classify(self, signal, interpolate=False, tinter=None):
        """ Classifies a signal. Returns a label 1 or -1

        signal : an m by n matrix
                 Last row should be the sampling times
        """
        if stl.satisfies(self.primitive, llt.SimpleModel(signal, interpolate, tinter)):
            if self.left is None:
                return 1
            else:
                return self.left.classify(signal, interpolate, tinter)
        else:
            if self.right is None:
                return -1
            else:
                return self.right.classify(signal, interpolate, tinter)

    def get_formula(self):
        """ Obtains an STL formula equivalent to this tree
        """
        left = self.primitive
        right = stl.STLNot(self.primitive)
        if self.left is not None:
            left = stl.STLAnd([self.primitive, self.left.get_formula()])
        if self.right is not None:
            return stl.STLOr([left, stl.STLAnd([right, self.right.get_formula()])])
        else:
            return left

    def add_signal(self, signal, label, interpolate=False, tinter=None):
        return self._add_signal(signal, label, np.Inf, interpolate, tinter)

    def _add_signal(self, signal, label, rho, interpolate=False, tinter=None):
        self.traces.add_traces([signal], [label])
        self.robustness = np.r_[self.robustness, rho]
        prim_rho = stl.robustness(
            self.primitive, llt.SimpleModel(signal, interpolate, tinter)
        )
        rho = np.amin([np.abs(prim_rho), rho])
        if prim_rho >= 0:
            if self.left is not None:
                return self.left._add_signal(signal, label, rho, interpolate, tinter)
            else:
                return self
        else:
            if self.right is not None:
                return self.right._add_signal(signal, label, rho, interpolate, tinter)
            else:
                return self

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value
        if value is not None:
            value.parent = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value
        if value is not None:
            value.parent = self

    def depth(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.depth()

    def pprint(self, tab=0):
        return _pprint(self, tab)


def _pprint(tree, tab=0):
    pad = " |" * tab + "-"
    if tree is None:
        return pad + "None\n"
    left = _pprint(tree.left, tab + 1)
    right = _pprint(tree.right, tab + 1)
    return "{}{} ({})\n{}{}".format(
        pad, str(tree.primitive), len(tree.traces), left, right
    )


def _pool_initializer():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class LLTInf(object):
    """ Obtains a decision tree that classifies the given labeled traces.

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

    TODO: Fix comments

    """

    def __init__(
        self,
        depth=1,
        primitive_factory=llt.make_llt_primitives,
        optimize_impurity=impurity.ext_inf_gain,
        stop_condition=None,
        redo_after_failed=1,
        optimizer_args=None,
        times=None,
        fallback_impurity=impurity.inf_gain,
    ):
        self.depth = depth
        self.primitive_factory = primitive_factory
        self.optimize_impurity = optimize_impurity
        self.fallback_impurity = fallback_impurity
        if stop_condition is None:
            self.stop_condition = [perfect_stop]
        else:
            self.stop_condition = stop_condition
        if optimizer_args is None:
            optimizer_args = {}
        self.optimizer_args = optimizer_args
        self.times = times
        self.interpolate = times is not None
        if self.interpolate and len(self.times) > 1:
            self.tinter = self.times[1] - self.times[0]
        else:
            self.tinter = None
        self.tree = None
        self.redo_after_failed = redo_after_failed
        self._partial_add = 0
        if "workers" not in self.optimizer_args:
            self.pool = Pool(initializer=_pool_initializer)

            def pool_map(func, iterable):
                try:
                    return self.pool.map_async(func, iterable).get(timeout=5)
                except KeyboardInterrupt:
                    self.pool.terminate()
                    self.pool.join()
                    raise KeyboardInterrupt()

            self.pool_map = pool_map
            self.optimizer_args["workers"] = self.pool_map

    def __del__(self):
        if hasattr(self, "pool"):
            self.pool.terminate()
            self.pool.join()

    def __exit__(self):
        if hasattr(self, "pool"):
            self.pool.terminate()
            self.pool.join()

    def fit(self, traces, disp=False):
        np.seterr(all="ignore")
        self.tree = self._lltinf(traces, None, self.depth, disp=disp)
        return self

    def fit_partial(self, traces, disp=False):
        if self.tree is None:
            return self.fit(traces, disp=disp)
        else:
            preds = self.predict(traces.signals)
            failed = set()
            for i in range(len(preds)):
                leaf = self.tree.add_signal(
                    traces.signals[i], traces.labels[i], self.interpolate, self.tinter
                )
                if preds[i] != traces.labels[i]:
                    failed.add(leaf)

            # logger.debug("Failed set: {}".format(failed))

            self._partial_add += len(failed)
            if self._partial_add // self.redo_after_failed > 0:
                # logger.debug("Redoing tree")
                self._partial_add = 0
                return self.fit(self.tree.traces, disp=disp)
            else:
                for leaf in failed:
                    # TODO don't redo whole node, only leaf
                    tree = self._lltinf(
                        leaf.traces,
                        leaf.robustness,
                        self.depth - leaf.depth(),
                        disp=disp,
                    )
                    old_tree = leaf.copy()
                    leaf.set_tree(tree)

                # FIXME only for perfect_stop
                preds = self.predict(traces.signals)
                if not np.array_equal(preds, traces.labels):
                    self._partial_add = 0
                    return self.fit(self.tree.traces, disp=disp)
                return self

    def predict(self, signals):
        if self.tree is not None:
            return np.array(
                [self.tree.classify(s, self.interpolate, self.tinter) for s in signals]
            )
        else:
            raise ValueError("Model not fit")

    def get_formula(self):
        if self.tree is not None:
            return self.tree.get_formula()
        else:
            raise ValueError("Model not fit")

    def _lltinf(self, traces, rho, depth, disp=False, override_impurity=None):
        """ Recursive call for the decision tree construction.

        See lltinf for information on similar arguments.

        rho : list of numerics
            List of robustness values for each trace up until the current node
        depth : integer
                Maximum depth to be reached. Decrements for each recursive call
        """
        # Stopping condition
        if any([stop(self, traces, rho, depth) for stop in self.stop_condition]):
            return None

        # Find primitive using impurity measure
        primitives = self.primitive_factory(traces.signals)
        if override_impurity is None:
            impurity = self.optimize_impurity
        else:
            impurity = override_impurity
        primitive, impurity = _find_best_primitive(
            traces,
            primitives,
            rho,
            impurity,
            disp,
            self.optimizer_args,
            times=self.times,
            interpolate=self.interpolate,
            tinter=self.tinter,
        )
        if disp:
            print("Best: {} ({})".format(primitive, impurity))

        # Classify using best primitive and split into groups
        prim_rho = [
            stl.robustness(primitive, llt.SimpleModel(s, self.interpolate, self.tinter))
            for s in traces.signals
        ]
        if rho is None:
            rho = [np.inf for i in traces.labels]
        tree = DTree(primitive, traces, rho)
        # [prim_rho, rho, signals, label]
        sat_, unsat_ = split_groups(
            list(zip(prim_rho, rho, *traces.as_list())), lambda x: x[0] >= 0
        )

        pure_wrong = all([t[3] <= 0 for t in sat_]) or all([t[3] >= 0 for t in unsat_])
        pure_right = all([t[3] >= 0 for t in sat_]) or all([t[3] <= 0 for t in unsat_])
        # Switch sat and unsat if labels are wrong. No need to negate prim rho since
        # we use it in absolute value later
        if pure_wrong or (
            not pure_right
            and (
                len([t for t in sat_ if t[3] >= 0])
                < len([t for t in unsat_ if t[3] >= 0])
            )
        ):
            sat_, unsat_ = unsat_, sat_
            tree.primitive = stl.STLNot(tree.primitive)

        # No further classification possible
        if len(sat_) == 0 or len(unsat_) == 0:
            logger.debug("No further classification possible")
            if override_impurity is None:
                logger.debug("Attempting to classify using impurity fallback")
                return self._lltinf(
                    traces,
                    rho,
                    depth,
                    disp=disp,
                    override_impurity=self.fallback_impurity,
                )
            else:
                return None

        # Redo data structures
        sat, unsat = [
            (Traces(*group[2:]), np.amin([np.abs(group[0]), group[1]], 0))
            for group in [list(zip(*sat_)), list(zip(*unsat_))]
        ]

        # Recursively build the tree
        tree.left = self._lltinf(sat[0], sat[1], depth - 1, disp=disp)
        tree.right = self._lltinf(unsat[0], unsat[1], depth - 1, disp=disp)

        return tree


def perfect_stop(lltinf, traces, rho, depth):
    """ Returns True if all traces are equally labeled.
    """
    return all([l > 0 for l in traces.labels]) or all([l <= 0 for l in traces.labels])


def depth_stop(lltinf, traces, rho, depth):
    """ Returns True if the maximum depth has been reached
    """
    return depth <= 0


def _find_best_primitive(
    traces,
    primitives,
    robustness,
    optimize_impurity,
    disp,
    optimizer_args,
    times,
    interpolate,
    tinter,
):
    # Parameters will be set for the copy of the primitive
    opt_prims = [
        impurity.optimize_impurity(
            traces,
            primitive.copy(),
            robustness,
            disp,
            optimizer_args,
            times,
            interpolate,
            tinter,
            impurity=optimize_impurity,
        )
        for primitive in primitives
    ]
    if disp:
        for p, imp in opt_prims:
            print("{} ({})".format(p, imp))

    return min(opt_prims, key=lambda x: x[1])
