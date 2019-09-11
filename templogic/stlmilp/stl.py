"""Module with STL definitions

Author: Francisco Penedo (franp@bu.edu)

"""

from abc import ABC, abstractmethod
import copy
from typing import Tuple, TypeVar, Iterable, Callable, Sequence, Generic, Union

import numpy as np
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

from templogic import util

import logging

__all__ = [
    "STLModel",
    "Signal",
    "STLTerm",
    "DisjTerm",
    "ConjTerm",
    "BooleanTerm",
    "TemporalTerm",
    "STLPred",
    "STLNot",
    "STLAnd",
    "STLOr",
    "STLAlways",
    "STLEventually",
    "STLNext",
    "robustness",
    "horizon",
    "satisfies",
    "is_negation_form",
    "perturb",
    "scale_time",
    "RobustnessTree",
    "stl_parser",
]

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


class Signal(Generic[U, T]):
    """Class for an observed signal.

    `U` is the type of a variable at a single time for the associated model

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

    bounds: Tuple[float, float]
    labels_at_t: Callable[["Signal[U, T]", float], Iterable[T]]

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
        self.bounds = bounds if bounds else (-1, 1)

    @property
    def labels(self) -> Labels:
        return self._labels

    @labels.setter
    def labels(self, value: Labels) -> None:
        self._labels = value
        try:
            if iter(self._labels):  # type: ignore
                setattr(self, "labels_at_t", self._label_list)
        except TypeError:
            setattr(self, "labels_at_t", self._label_fun)

    def _label_fun(self, t: float) -> Iterable[T]:
        return self.labels(t)  # type: ignore

    def _label_list(self, t: float) -> Iterable[T]:
        return [l(t) for l in self.labels]  # type: ignore

    def _get_vs(self, model: STLModel[T, U], t: float) -> Sequence[U]:
        return [model.getVarByName(l) for l in self.labels_at_t(t)]

    def perturb(self, eps: Callable[["Signal[U, T]"], U]) -> "Signal[U, T]":
        return self

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


class STLTerm(ABC):
    """Any STL term is derived from this class

    This class implements general functions to traverse the complete formula and
    delegates the concrete implementation to specific terms
    """

    args: Sequence["STLTerm"]

    def __init__(self, args: Sequence["STLTerm"]) -> None:
        self.args = args

    def map(self, f: Callable[["STLTerm"], "STLTerm"]) -> "STLTerm":
        args = [arg.map(f) for arg in self.args]
        obj = f(self)
        obj.args = args
        return obj

    def fold(self, g: Callable[["STLTerm", Iterable[T]], T]) -> T:
        return g(self, [arg.fold(g) for arg in self.args])

    def temporalFold(
        self, g: Callable[["STLTerm", float, Iterable[T]], T], t: float
    ) -> T:
        return g(self, t, [arg.temporalFold(g, t) for arg in self.args])

    def copy(self) -> "STLTerm":
        return copy.deepcopy(self)

    def shallowcopy(self) -> "STLTerm":
        return copy.copy(self)

    @abstractmethod
    def __str__(self) -> str:
        pass


class DisjTerm(STLTerm):
    pass


class ConjTerm(STLTerm):
    pass


class BooleanTerm(STLTerm):
    def __init__(self, args: Sequence[STLTerm]):
        if len(args) == 0:
            raise ValueError(
                f"{self.__class__.__name__} must have at least one argument"
            )
        super().__init__(args)


class TemporalTerm(STLTerm):
    bounds: Tuple[float, float]

    def __init__(self, bounds: Tuple[float, float], arg: STLTerm):
        super().__init__([arg])
        self.bounds = bounds

    def temporalFold(
        self, g: Callable[["STLTerm", float, Iterable[T]], T], t: float
    ) -> T:
        return g(
            self,
            t,
            [
                self.args[0].temporalFold(g, t + j)
                for j in np.arange(self.bounds[0], self.bounds[1] + 1)
            ],
        )


class STLPred(ConjTerm, STLTerm):

    signal: Signal

    def __init__(self, arg: Signal):
        super().__init__([])
        self.signal = arg

    def __str__(self) -> str:
        return f"({self.signal})"


class STLNot(ConjTerm, BooleanTerm, STLTerm):
    def __init__(self, arg: STLTerm):
        super().__init__([arg])

    def __str__(self) -> str:
        return "~ {}".format(str(self.args[0]))


class STLAnd(BooleanTerm, ConjTerm, STLTerm):
    def __str__(self) -> str:
        return "({})".format(" & ".join(str(arg) for arg in self.args))


class STLOr(BooleanTerm, DisjTerm, STLTerm):
    def __str__(self) -> str:
        return "({})".format(" | ".join(str(arg) for arg in self.args))


class STLEventually(TemporalTerm, DisjTerm, STLTerm):
    def __str__(self) -> str:
        return f"F_[{self.bounds[0]:.2f}, {self.bounds[1]:.2f}] {self.args[0]}"


class STLAlways(TemporalTerm, ConjTerm, STLTerm):
    def __str__(self) -> str:
        return f"G_[{self.bounds[0]:.2f}, {self.bounds[1]:.2f}] {self.args[0]}"


class STLNext(ConjTerm, STLTerm):
    def __init__(self, arg: STLTerm):
        super().__init__([arg])

    def temporalFold(
        self, g: Callable[["STLTerm", float, Iterable[T]], T], t: float
    ) -> T:
        return g(self, t, [self.args[0].temporalFold(g, t + 1)])

    def __str__(self) -> str:
        return f"O {self.args[0]}"


def robustness(formula: STLTerm, model: STLModel, t: float = 0) -> float:
    def _rob(term: STLTerm, t: float, robs: Iterable[float]) -> float:
        if isinstance(term, STLPred):
            return term.signal.signal(model, t)
        elif isinstance(term, STLNot):
            return -list(robs)[0]
        elif isinstance(term, ConjTerm):
            return min(robs)
        elif isinstance(term, DisjTerm):
            return max(robs)
        else:
            raise Exception(f"Non exhaustive pattern matching {term.__class__}")

    f = scale_time(formula, model.tinter)
    return f.temporalFold(_rob, t)


def horizon(formula: STLTerm) -> float:
    def _hor(term: STLTerm, hors: Iterable[float]) -> float:
        if isinstance(term, TemporalTerm):
            return term.bounds[1] + list(hors)[0]
        elif isinstance(term, STLNext):
            return 1 + list(hors)[0]
        elif isinstance(term, STLPred):
            return 0
        else:
            return max(hors)

    return formula.fold(_hor)


def satisfies(formula: STLTerm, model: STLModel, t: float = 0) -> bool:
    """Checks if a model satisfies a formula at some time.

    Satisfaction is defined in this function as robustness >= 0.

    formula : Formula
    model : a model as defined in Signal
    t : numeric
        The time
    """
    return robustness(formula, model, t) >= 0


def is_negation_form(f: STLTerm) -> bool:
    def _neg_form(term: STLTerm, values: Iterable[bool]) -> bool:
        if isinstance(term, STLNot):
            return isinstance(term.args[0], STLPred)
        return all(values)

    return f.fold(_neg_form)


def perturb(f: STLTerm, eps: Callable[["Signal[U, T]"], U]) -> STLTerm:
    if not is_negation_form(f):
        raise Exception("Formula not in negation form")

    def _perturb(term: STLTerm) -> STLTerm:
        if isinstance(term, STLPred):
            term.signal.perturb(eps)
        return term

    return f.map(_perturb)


def scale_time(f: STLTerm, dt: float) -> STLTerm:
    """Transforms a formula in continuous time to discrete time

    Substitutes the time bounds in a :class:`stlmilp.stl.STLTerm` from
    continuous time to discrete time with time interval `dt`

    Parameters
    ----------
    formula : :class:`stlmilp.stl.STLTerm`
    dt : float

    Returns
    -------
    None

    """

    def _scale(term: STLTerm) -> STLTerm:
        obj = term.shallowcopy()
        if isinstance(obj, TemporalTerm):
            obj.bounds = tuple([int(b / dt) for b in obj.bounds])  # type: ignore
        return obj

    return f.map(_scale)


TreeData = Tuple[float, int]


class RobustnessTree(util.Tree[TreeData, "RobustnessTree"]):
    @property
    def robustness(self) -> float:
        return self._data[0]

    @property
    def index(self) -> int:
        return self._data[1]


def robustness_tree(formula: STLTerm, model: STLModel, t: float = 0) -> RobustnessTree:
    def _rob(term: STLTerm, t: float, robs: Iterable[RobustnessTree]) -> RobustnessTree:
        robs = list(robs)
        if isinstance(term, STLPred):
            return RobustnessTree((term.signal.signal(model, t), 0), [])
        elif isinstance(term, STLNot):
            child = robs[0]
            return RobustnessTree((-child.robustness, 0), [child])
        elif isinstance(term, ConjTerm):
            i = np.argmin([r.robustness for r in robs])
            return RobustnessTree((robs[i].robustness, i), robs)
        elif isinstance(term, DisjTerm):
            i = np.argmax([r.robustness for r in robs])
            return RobustnessTree((robs[i].robustness, i), robs)
        else:
            raise Exception(f"Non exhaustive pattern matching {term.__class__}")

    f = scale_time(formula, model.tinter)
    return f.temporalFold(_rob, t)


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

    expr : a parser, optional, defaults to r'\\w+'
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
        num = int_parser()
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
