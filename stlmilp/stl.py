"""Module with STL definitions

Author: Francisco Penedo (franp@bu.edu)

"""

import operator
import copy
import numpy as np
from pyparsing import (
    Word,
    Suppress,
    Optional,
    Combine,
    nums,
    Literal,
    Forward,
    delimitedList,
    alphanums,
    Keyword,
    Group,
)

# Operator constants
EXPR = 0
NOT = 1
AND = 2
OR = 3
NEXT = 4
ALWAYS = 5
EVENTUALLY = 6

LE = operator.le
GT = operator.gt


class Signal(object):
    """Class for an observed signal.

    Example: let y(t) = x1(t) + x2(t) be an observed (or secondary) signal for a
    model with primary signals x1 and x2. We can define this signal as follows:

    >>> x = [[1,2,3], [3,4,5]]
    >>> class Model():
    >>>     def getVarByName(self, i):
    >>>         return x[i[0]][i[1]]
    >>> labels = [lambda t: (0, t), lambda t: (1, t)]
    >>> f = lambda x, y: x + y
    >>> y = Signal(labels, f)
    >>> signal(Model(), 1)
    6

    """

    def __init__(self, labels, f, bounds=None):
        """Class for an observed signal.

        labels : array of functions
                 Functions that return the name of the primary signals at any
                 given time needed for this observed signal.
        f : function
            Function of the primary signals. Arity should be equal to the length
            of labels.
        """

        self.labels = labels
        self.f = f
        self.bounds = bounds if bounds else [-1, 1]

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value
        try:
            if len(self.labels) >= 0:
                self.get_vs = self._get_vs_list
        except TypeError:
            self.get_vs = self._get_vs_fun

    def _get_vs_list(self, model, t):
        return [model.getVarByName(l(t)) for l in self.labels]

    def _get_vs_fun(self, model, t):
        return [model.getVarByName(l) for l in self.labels(t)]

    def signal(self, model, t):
        """Obtain the observed signal at time t for the given model.

        model : object with a getVarByName(self, signal_t) method
                The model containing the time series for the primary signals.
                The method getVarByName should accept objects returned by the
                functions in the labels parameter to __init__ and return the
                value of the signal at the given time
        t : numeric
            The time
        """

        vs = self.get_vs(model, t)
        # TODO Get rid of any
        if any(var is None for var in vs):
            raise Exception(
                (
                    "Couldn't find all variables in model.\n" "Labels: {}\n" "vs: {}"
                ).format([l(t) for l in self.labels], vs)
            )
        else:
            return self.f(vs)

    def __str__(self):
        return "EXP"

    def __repr__(self):
        return self.__str__()


class Formula(object):
    """An STL formula.

    """

    def __init__(self, operator, args, bounds=None):
        """An STL formula

        operator : one of EXPR, AND, OR, NOT, ALWAYS, EVENTUALLY, NEXT
        args : either a list of Formulas or a Signal
               If operator is EXPR, this is the signal corresponding to the
               atomic formula g(t) > 0. Otherwise, the list of arguments of the
               operator (1 for NOT, and the temporal operators, 2 or more for
               AND and OR).
        bounds : list of two numerics, optional
                 The bounds of the temporal operator. Defaults to [0, 0]
        """
        self._op = operator
        self._args = args
        if bounds is None:
            bounds = [0, 0]
        self._bounds = bounds

    def _hexpr(self):
        return 0

    def _hnot(self):
        return 0

    def _hand(self):
        # return max(map(lambda f: f.horizon(), self.args))
        return max([f.horizon() for f in self.args])

    def _hor(self):
        return self._hand()

    def _halways(self):
        return self.bounds[1] + self.args[0].horizon()

    def _hnext(self):
        return 1 + self.args[0].horizon()

    def _heventually(self):
        return self._halways()

    def horizon(self):
        """Computes the time horizon of the formula
        """
        return {
            EXPR: self._hexpr,
            NOT: self._hnot,
            AND: self._hand,
            OR: self._hor,
            NEXT: self._hnext,
            ALWAYS: self._halways,
            EVENTUALLY: self._heventually,
        }[self.op]()

    def copy(self):
        return copy.deepcopy(self)

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, value):
        self._op = value

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

    def __str__(self):
        if self.op == EXPR:
            string = "(%s)" % str(self.args[0])
        elif self.op == NOT:
            string = "~ %s" % str(self.args[0])
        elif self.op == AND:
            string = "(%s)" % " & ".join([str(arg) for arg in self.args])
        elif self.op == OR:
            string = "(%s)" % " | ".join([str(arg) for arg in self.args])
        elif self.op == NEXT:
            string = "O %s" % str(self.args[0])
        elif self.op == ALWAYS:
            string = "G_[%.2f, %.2f] %s" % (
                self.bounds[0],
                self.bounds[1],
                str(self.args[0]),
            )
        elif self.op == EVENTUALLY:
            string = "F_[%.2f, %.2f] %s" % (
                self.bounds[0],
                self.bounds[1],
                str(self.args[0]),
            )
        else:
            raise Exception("Unrecognized op code: {}".format(self.op))

        return string

    def __repr__(self):
        return self.__str__()


def perturb(f, eps):
    if f.op == EXPR:
        f.args[0].perturb(eps)
    elif f.op == NOT:
        if f.args[0].op != EXPR:
            raise Exception("Formula not in negation form")
        else:
            perturb(f.args[0], eps)
    else:
        for arg in f.args:
            perturb(arg, eps)


def scale_time(formula, dt):
    """Transforms a formula in continuous time to discrete time

    Substitutes the time bounds in a :class:`stlmilp.stl.Formula` from
    continuous time to discrete time with time interval `dt`

    Parameters
    ----------
    formula : :class:`stlmilp.stl.Formula`
    dt : float

    Returns
    -------
    None

    """
    formula.bounds = [int(b / dt) for b in formula.bounds]
    for arg in formula.args:
        if arg.op != EXPR:
            scale_time(arg, dt)


def score(formula, model, ops, t=0):
    mmap, mreduce = ops[formula.op]
    return mreduce(mmap(formula.args, formula.bounds, model, t, ops))


def expr_map(args, bounds, model, t, ops):
    return [args[0].signal(model, t)]


def boolean_map(args, bounds, model, t, ops):
    return [score(arg, model, ops, t) for arg in args]


def next_map(args, bounds, model, t, ops):
    return [score(args[0], model, ops, t + model.tinter)]


def temp_map(args, bounds, model, t, ops):
    return [
        score(args[0], model, ops, t + j)
        for j in np.arange(bounds[0], bounds[1] + model.tinter, model.tinter)
    ]


def identity(xs):
    return xs[0]


def neg(xs):
    return -xs[0]


STL_ROBUSTNESS_OPS = {
    EXPR: [expr_map, identity],
    NOT: [boolean_map, neg],
    AND: [boolean_map, min],
    OR: [boolean_map, max],
    NEXT: [next_map, identity],
    ALWAYS: [temp_map, min],
    EVENTUALLY: [temp_map, max],
}


def robustness(formula, model, t=0):
    return score(formula, model, STL_ROBUSTNESS_OPS, t)


class RobustnessTree(object):
    def __init__(self, robustness, index, children):
        self.robustness = robustness
        self.index = index
        self.children = children

    @classmethod
    def expr_map(cls, args, bounds, model, t, ops):
        return [cls(expr_map(args, bounds, model, t, ops)[0], 0, [])]

    @classmethod
    def neg(cls, xs):
        return cls(-xs[0].robustness, 0, xs)

    @classmethod
    def _minmax(cls, xs, op):
        i = op([x.robustness for x in xs])
        return cls(xs[i].robustness, i, xs)

    @classmethod
    def min(cls, xs):
        return cls._minmax(xs, np.argmin)

    @classmethod
    def max(cls, xs):
        return cls._minmax(xs, np.argmax)

    def pprint(self, tab=0):
        return _pprint(self, tab)


def _pprint(tree, tab=0):
    pad = " |" * tab + "-"
    children = [_pprint(child, tab + 1) for child in tree.children]
    return "{}r = {} ({})\n{}".format(
        pad, tree.robustness, tree.index, "".join(children)
    )


ROBUSTNESS_TREE_OPS = {
    EXPR: [RobustnessTree.expr_map, identity],
    NOT: [boolean_map, RobustnessTree.neg],
    AND: [boolean_map, RobustnessTree.min],
    OR: [boolean_map, RobustnessTree.max],
    NEXT: [next_map, identity],
    ALWAYS: [temp_map, RobustnessTree.min],
    EVENTUALLY: [temp_map, RobustnessTree.max],
}


def robustness_tree(formula, model, t=0):
    return score(formula, model, ROBUSTNESS_TREE_OPS, t)


def satisfies(formula, model, t=0):
    """Checks if a model satisfies a formula at some time.

    Satisfaction is defined in this function as robustness >= 0.

    formula : Formula
    model : a model as defined in Signal
    t : numeric
        The time
    """
    return robustness(formula, model, t) >= 0


# parser


def num_parser():
    """A floating point number parser
    """
    T_DOT = Literal(".")
    T_MIN = Literal("-")
    T_PLU = Literal("+")
    T_EXP = Literal("e")
    num = Combine(
        Optional(T_MIN)
        + Word(nums)
        + Optional(T_DOT + Word(nums))
        + Optional(T_EXP + Optional(T_MIN ^ T_PLU) + Word(nums))
    )
    num = num.setParseAction(lambda t: float(t[0]))
    return num


def int_parser():
    T_MIN = Literal("-")
    num = Combine(Optional(T_MIN) + Word(nums))
    num = num.setParseAction(lambda t: int(t[0]))
    return num


def stl_parser(expr=None, float_bounds=True):
    """Builds an stl parser using the given expression parser.

    The STL grammar used is the following:

        form ::= ( expr )
               | "~" form
               | ( and_list )
               | ( or_list )
               | op "_" interval form
        and_list ::= form "^" form
                   | form "^" and_list
        or_list ::= form "v" form
                   | form "v" or_list
        op ::= "G" | "F"
        interval ::= [ num "," num ]

    where num is a floating point number

    expr : a parser, optional, defaults to r'\w+'
           An expression parser.
    """
    if not expr:
        expr = Word(alphanums)

    T_GLOB = Keyword("G", alphanums)
    T_FUT = Keyword("F", alphanums)
    T_NEXT = Keyword("O", alphanums)
    T_LPAR, T_RPAR, T_LBRK, T_RBRK, T_UND, T_COM, T_TILD = map(Suppress, "()[]_,~")
    if float_bounds:
        num = num_parser()
    else:
        num = Word(nums).setParseAction(lambda t: int(t[0]))
    interval = Group(T_LBRK + num + T_COM + num + T_RBRK)

    form = Forward()

    form_not = T_TILD + form
    form_and = T_LPAR + delimitedList(form, "&") + T_RPAR
    form_or = T_LPAR + delimitedList(form, "|") + T_RPAR
    form_expr = T_LPAR + expr + T_RPAR
    form_next = T_NEXT + form
    form_alw = T_GLOB + T_UND + interval + form
    form_fut = T_FUT + T_UND + interval + form

    form << (
        form_expr ^ form_not ^ form_and ^ form_or ^ form_next ^ form_alw ^ form_fut
    )

    form_expr.setParseAction(lambda t: Formula(EXPR, [t[0]]))
    form_not.setParseAction(lambda t: Formula(NOT, [t[0]]))
    form_and.setParseAction(lambda t: Formula(AND, list(t)))
    form_or.setParseAction(lambda t: Formula(OR, list(t)))
    form_next.setParseAction(lambda t: Formula(NEXT, [t[1]]))
    form_alw.setParseAction(lambda t: Formula(ALWAYS, [t[2]], bounds=list(t[1])))
    form_fut.setParseAction(lambda t: Formula(EVENTUALLY, [t[2]], bounds=list(t[1])))

    return form
