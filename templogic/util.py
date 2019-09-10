from bisect import bisect_right
import logging
import copy
from typing import Callable, Generic, Iterable, List, Optional, Sequence, TypeVar, cast
import pickle
from abc import ABC, abstractmethod

import attr
import numpy as np


logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U", bound="Tree")


class Tree(Generic[T, U]):
    """A tree data structure
    """

    _data: T
    _children: List[Optional[U]]
    parent: Optional[U]

    def __init__(self, data: T, children: Iterable[Optional[U]]) -> None:
        self.data = data
        self.children = list(children)
        self.parent = None

    def get_child(self, idx: int) -> Optional[U]:
        return self._children[idx]

    def set_child(self, idx: int, tree: Optional[U]) -> None:
        self._children[idx] = tree
        if tree is not None:
            tree.parent = self

    @property
    def children(self) -> List[Optional[U]]:
        return self._children

    @children.setter
    def children(self, value: Sequence[Optional[U]]) -> None:
        self._children = list(value)
        for child in self._children:
            if child is not None:
                child.parent = self

    @property
    def data(self) -> T:
        return self._data

    @data.setter
    def data(self, value: T) -> None:
        self._data = value

    def level(self) -> int:
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.level()

    def depth(self) -> int:
        if self.children is None or all([child is None for child in self._children]):
            return 0
        else:
            return 1 + max(
                [child.depth() for child in self._children if child is not None]
            )

    def foldl(self, f: Callable[[S, T], S], z: S) -> S:
        res = z
        l = []
        l.append(self)
        while len(l) > 0:
            t = l.pop(0)
            res = f(res, t.data)
            l.extend(c for c in t.children if c is not None)

        return res

    def flatten(self) -> List[T]:
        def append(l, a):
            l.append(a)
            return l

        return self.foldl(append, [])

    def pprint(self, tab: int = 0) -> str:
        return _tree_pprint(self, tab)

    def deep_copy(self) -> U:
        return cast(U, copy.deepcopy(self))

    def copy(self) -> U:
        return cast(U, copy.copy(self))

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return self.__str__()


def _tree_pprint(tree: Tree, tab: int = 0) -> str:
    pad = " |" * tab + "-"
    if tree is None:
        return pad + "None\n"
    children = [
        _tree_pprint(child, tab + 1) for child in tree.children if child is not None
    ]
    return "{}{}\n{}".format(pad, str(tree), "".join(children))


def split_groups(l, group):
    """ Splits a list according to a binary grouping function.

    Returns the positive group first

    l : a list
    group : a function from elements of l to boolean
    """
    p = [x for x in l if group(x)]
    n = [x for x in l if not group(x)]
    return p, n


def round_t(t: float, times: Optional[Sequence[float]]) -> float:
    if times is None:
        return t
    else:
        i = bisect_right(times, t) - 1
        return times[i]


class Classifier(ABC):
    @abstractmethod
    def classify(self, trace: np.ndarray) -> int:
        pass


@attr.s(auto_attribs=True)
class CVResult:
    miss_mean: float
    miss_std: float
    missrates: Sequence[float]
    classifiers: Sequence[Classifier]


def cross_validation(
    data: np.ndarray,
    labels: Iterable[int],
    learn: Callable[[np.ndarray, Iterable[int]], Classifier],
    k: int = 10,
    save: Optional[str] = None,
    disp: bool = False,
) -> CVResult:
    """ Performs a k-fold cross validation test.

    data : a list of labeled traces
           The input data for the cross validation test. It must be a list of
           pairs [trace, label], where the trace is an m by n matrix with the
           last row being the sampling times and the label is 1 or -1.
    learn : a function from data to a classifier
            The learning function. Must accept as a parameter a subset of the
            data and return a classifier. A classifier must be an object with
            a method classify(trace), where trace is defined as in the data
            argument.
    k : integer, optional, defaults to 10
        The number of folds
    save : string, optional
           If specified, the name of a file to save the permutation used to
           split the data.
    disp : boolean, optional, defaults to False
           Toggles the output of debugging information

    """
    if k > len(data):
        raise AttributeError("Fold number should not exceed size of dataset")
    p = np.random.permutation(len(data))
    if save is not None:
        with open(save, "wb") as out:
            pickle.dump(p.tolist(), out)

    perm = list(zip(np.array(data)[p], np.array(labels)[p]))
    n = len(data) // k
    folds = [perm[i * n : (i + 1) * n] for i in range(k)]
    if len(data) % k != 0:
        folds[-1] = np.append(folds[-1], perm[k * n :], axis=0)

    missrates = []
    classifiers = []
    for i in range(k):
        lfolds = folds[:i] + folds[(i + 1) :]
        ldata, llabels = list(zip(*[x for fold in lfolds for x in fold]))
        classifier = learn(ldata, llabels)
        fold_data, fold_labels = list(zip(*folds[i]))
        missrates.append(missrate(fold_data, fold_labels, classifier))
        classifiers.append(classifier)
        if disp:
            print(f"Cross validation step {i}")
            print(f"Miss: {missrates[i]}")
            print(str(classifier))

    return CVResult(np.mean(missrates), np.std(missrates), missrates, classifiers)


def missrate(data: np.ndarray, labels: Iterable[int], classifier: Classifier) -> float:
    """ Obtains the missrate of a classifier on a given validation set.

    validate : a list of labeled traces
               A validation set. See cross_validation for a description of the
               format
    classifier : an object with a classify method
                 The classifier. See cross_validation for a description

    """
    labels_ = np.array(labels)
    test = np.array([classifier.classify(x) for x in data])
    return np.count_nonzero(labels_ - test) / float(len(labels_))
