"""Module with STL definitions

Author: Francisco Penedo (franp@bu.edu)

"""

from abc import ABC, abstractmethod
import copy
from typing import Tuple, TypeVar, Iterable, Callable, Sequence, Generic, Union
from typing_extensions import Protocol

import numpy as np  # type: ignore
from pyparsing import (  # type: ignore
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

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")


class STLModel(ABC, Generic[T, U]):
    """Model for STL logic

    `T` is the type of a label
    `U` is the type of a model variable at a single time
    """

    tinter: float

    @abstractmethod
    def getVarByName(self, label_t: T) -> U:
        """Returns the value of the variable represented by `label_t`

        A concrete class will define the type of labels accepted. A label should
        encode at least a time. For example, a one-dimensional time series will have
        T == float
        """
        pass


Labels = Union[Callable[[float], Iterable[T]], Iterable[Callable[[float], T]]]


class Signal(Generic[U]):
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

    def __init__(
        self,
        labels: Labels,
        f: Callable[[Sequence[U]], float],
        bounds: Tuple[float, float] = None,
    ) -> None:
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
    def labels(self) -> Labels:
        return self._labels

    @labels.setter
    def labels(self, value: Labels) -> None:
        self._labels = value
        try:
            if iter(self.labels):  # type: ignore
                self.labels_at_t = self._label_list
        except TypeError:
            self.labels_at_t = self._label_fun

    def _label_fun(self, t: float) -> Iterable[T]:
        return self.labels(t)  # type: ignore

    def _label_list(self, t: float) -> Iterable[T]:
        return [l(t) for l in self.labels]  # type: ignore

    def _get_vs(self, model: STLModel[T, U], t: float) -> Sequence[U]:
        return [model.getVarByName(l) for l in self.labels_at_t(t)]

    def signal(self, model: STLModel[T, U], t: float) -> float:
        """Obtain the observed signal at time t for the given model.

        model : object with a getVarByName(self, signal_t) method
                The model containing the time series for the primary signals.
                The method getVarByName should accept objects returned by the
                functions in the labels parameter to __init__ and return the
                value of the signal at the given time
        t : numeric
            The time
        """

        vs = self._get_vs(model, t)
        # TODO Get rid of any
        if any(var is None for var in vs):
            raise Exception("Couldn't find all variables in model.")
        else:
            return self.f(vs)

    def __str__(self) -> str:
        return "EXP"

    def __repr__(self) -> str:
        return self.__str__()


class MMap(Protocol[T]):
    def __call__(
        self,
        model: STLModel,
        t: float,
        mmap: Callable[["STLTerm"], "MMap[T]"],
        mreduce: Callable[["STLTerm"], "MReduce[T]"],
    ) -> Iterable[T]:
        pass


class MReduce(Protocol[T]):
    def __call__(self, rhos: Iterable[T]) -> T:
        pass


MMapGet = Callable[["STLTerm"], MMap[T]]
MReduceGet = Callable[["STLTerm"], MReduce[T]]


class STLTerm(ABC):
    """Any STL term is derived from this class

    This class implements general functions to traverse the complete formula and
    delegates the concrete implementation to specific terms
    """

    def score(
        self, model: STLModel, t: float, mmap: MMapGet[T], mreduce: MReduceGet[T]
    ) -> T:
        return mreduce(self)(mmap(self)(model, t, mmap, mreduce))

    def horizon(self) -> int:
        return self.score(  # type: ignore # Passing None to score
            None, 0, lambda obj: obj.horizon_map, lambda obj: obj.horizon_reduce
        )

    def robustness(self, model: STLModel, t: float = 0) -> float:
        return self.score(model, t, lambda obj: obj.rho_map, lambda obj: obj.rho_reduce)

    @abstractmethod
    def rho_map(
        self,
        model: STLModel,
        t: float,
        mmap: MMapGet[float],
        mreduce: MReduceGet[float],
    ) -> Iterable[float]:
        pass

    def horizon_map(
        self,
        model: STLModel,
        t: float,
        mmap: MMapGet[float],
        mreduce: MReduceGet[float],
    ) -> Iterable[float]:
        return [0]

    @abstractmethod
    def rho_reduce(self, rhos: Iterable[float]) -> float:
        pass

    def horizon_reduce(self, rhos: Iterable[float]) -> float:
        return max(rhos)

    def copy(self) -> "STLTerm":
        return copy.copy(self)

    @abstractmethod
    def __str__(self) -> str:
        pass


class DisjTerm(object):
    def rho_reduce(self, rhos: Iterable[float]) -> float:
        return max(rhos)


class ConjTerm(object):
    def rho_reduce(self, rhos: Iterable[float]) -> float:
        return min(rhos)


class BooleanTerm(object):
    args: Sequence[STLTerm]

    def rho_map(
        self,
        model: STLModel,
        t: float,
        mmap: MMapGet[float],
        mreduce: MReduceGet[float],
    ) -> Iterable[float]:
        return [arg.score(model, t, mmap, mreduce) for arg in self.args]

    def horizon_map(self, model, t, mmap, mreduce) -> Iterable[float]:
        return self.rho_map(model, t, mmap, mreduce)


class TemporalTerm(object):
    arg: STLTerm
    bounds: Tuple[float, float]

    def rho_map(
        self,
        model: STLModel,
        t: float,
        mmap: MMapGet[float],
        mreduce: MReduceGet[float],
    ) -> Iterable[float]:
        return [
            self.arg.score(model, t + j, mmap, mreduce)
            for j in np.arange(self.bounds[0], self.bounds[1] + model.tinter)
        ]

    def horizon_map(self, model, t, mmap, mreduce) -> Iterable[float]:
        return [self.bounds[1] + self.arg.score(model, t, mmap, mreduce)]


class STLPred(ConjTerm, STLTerm):
    def __init__(self, arg: Signal):
        self.arg: Signal = arg

    def rho_map(
        self,
        model: STLModel,
        t: float,
        mmap: MMapGet[float],
        mreduce: MReduceGet[float],
    ) -> Iterable[float]:
        return [self.arg.signal(model, t)]

    def __str__(self) -> str:
        return "(EXP)"


class STLNot(ConjTerm, BooleanTerm, STLTerm):
    def __init__(self, arg: STLTerm):
        self.args: Sequence[STLTerm] = [arg]

    def rho_reduce(self, rhos: Iterable[float]) -> float:
        return -super(STLNot, self).rho_reduce(rhos)

    def __str__(self) -> str:
        return "~ {}".format(str(self.args[0]))


class STLAnd(BooleanTerm, ConjTerm, STLTerm):
    def __init__(self, args: Sequence[STLTerm]):
        if len(args) == 0:
            raise ValueError("STLAnd must have at least one argument")
        self.args = args

    def __str__(self) -> str:
        return "({})".format(" & ".join(str(arg) for arg in self.args))


class STLOr(BooleanTerm, DisjTerm, STLTerm):
    def __init__(self, args: Sequence[STLTerm]):
        if len(args) == 0:
            raise ValueError("STLOr must have at least one argument")
        self.args = args

    def __str__(self) -> str:
        return "({})".format(" | ".join(str(arg) for arg in self.args))


class STLEventually(TemporalTerm, DisjTerm, STLTerm):
    def __init__(self, bounds: Tuple[float, float], arg: STLTerm):
        self.arg = arg
        self.bounds = bounds

    def __str__(self) -> str:
        return f"F_[{self.bounds[0]:.2f}, {self.bounds[1]:.2f}] {self.arg}"


class STLAlways(TemporalTerm, ConjTerm, STLTerm):
    def __init__(self, bounds: Tuple[float, float], arg: STLTerm):
        self.arg = arg
        self.bounds = bounds

    def __str__(self) -> str:
        return f"G_[{self.bounds[0]:.2f}, {self.bounds[1]:.2f}] {self.arg}"


class STLNext(ConjTerm, STLTerm):
    def __init__(self, arg: STLTerm):
        self.arg = arg

    def rho_map(
        self,
        model: STLModel,
        t: float,
        mmap: MMapGet[float],
        mreduce: MReduceGet[float],
    ) -> Iterable[float]:
        return [self.arg.score(model, t + model.tinter, mmap, mreduce)]

    def horizon_map(self, model, t, mmap, mreduce) -> Iterable[float]:
        return [1 + self.arg.score(model, t, mmap, mreduce)]

    def __str__(self) -> str:
        return f"O {self.arg}"


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

    form_expr.setParseAction(lambda t: STLPred(t[0]))
    form_not.setParseAction(lambda t: STLNot(t[0]))
    form_and.setParseAction(lambda t: STLAnd(list(t)))
    form_or.setParseAction(lambda t: STLOr(list(t)))
    form_next.setParseAction(lambda t: STLNext(t[1]))
    form_alw.setParseAction(lambda t: STLAlways(arg=t[2], bounds=tuple(t[1])))
    form_fut.setParseAction(lambda t: STLEventually(arg=t[2], bounds=tuple(t[1])))

    return form


# def perturb(f, eps):
#     if f.op == EXPR:
#         f.args[0].perturb(eps)
#     elif f.op == NOT:
#         if f.args[0].op != EXPR:
#             raise Exception("Formula not in negation form")
#         else:
#             perturb(f.args[0], eps)
#     else:
#         for arg in f.args:
#             perturb(arg, eps)
#
#
# def scale_time(formula, dt):
#     """Transforms a formula in continuous time to discrete time
#
#     Substitutes the time bounds in a :class:`stlmilp.stl.Formula` from
#     continuous time to discrete time with time interval `dt`
#
#     Parameters
#     ----------
#     formula : :class:`stlmilp.stl.Formula`
#     dt : float
#
#     Returns
#     -------
#     None
#
#     """
#     formula.bounds = [int(b / dt) for b in formula.bounds]
#     for arg in formula.args:
#         if arg.op != EXPR:
#             scale_time(arg, dt)
#
#
# def score(formula, model, ops, t=0):
#     mmap, mreduce = ops[formula.op]
#     return mreduce(mmap(formula.args, formula.bounds, model, t, ops))
#
#
# class RobustnessTree(object):
#     def __init__(self, robustness, index, children):
#         self.robustness = robustness
#         self.index = index
#         self.children = children
#
#     @classmethod
#     def expr_map(cls, args, bounds, model, t, ops):
#         return [cls(expr_map(args, bounds, model, t, ops)[0], 0, [])]
#
#     @classmethod
#     def neg(cls, xs):
#         return cls(-xs[0].robustness, 0, xs)
#
#     @classmethod
#     def _minmax(cls, xs, op):
#         i = op([x.robustness for x in xs])
#         return cls(xs[i].robustness, i, xs)
#
#     @classmethod
#     def min(cls, xs):
#         return cls._minmax(xs, np.argmin)
#
#     @classmethod
#     def max(cls, xs):
#         return cls._minmax(xs, np.argmax)
#
#     def pprint(self, tab=0):
#         return _pprint(self, tab)
#
#
# def _pprint(tree, tab=0):
#     pad = " |" * tab + "-"
#     children = [_pprint(child, tab + 1) for child in tree.children]
#     return "{}r = {} ({})\n{}".format(
#         pad, tree.robustness, tree.index, "".join(children)
#     )
#
#
# ROBUSTNESS_TREE_OPS = {
#     EXPR: [RobustnessTree.expr_map, identity],
#     NOT: [boolean_map, RobustnessTree.neg],
#     AND: [boolean_map, RobustnessTree.min],
#     OR: [boolean_map, RobustnessTree.max],
#     NEXT: [next_map, identity],
#     ALWAYS: [temp_map, RobustnessTree.min],
#     EVENTUALLY: [temp_map, RobustnessTree.max],
# }
#
#
# def robustness_tree(formula, model, t=0):
#     return score(formula, model, ROBUSTNESS_TREE_OPS, t)
#
#
# def satisfies(formula, model, t=0):
#     """Checks if a model satisfies a formula at some time.
#
#     Satisfaction is defined in this function as robustness >= 0.
#
#     formula : Formula
#     model : a model as defined in Signal
#     t : numeric
#         The time
#     """
#     return robustness(formula, model, t) >= 0
#
#
# parser
