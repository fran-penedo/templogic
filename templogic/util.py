import logging
from typing import Callable, Generic, Iterable, List, Optional, Sequence, TypeVar


logger = logging.getLogger(__name__)

T = TypeVar("T")
S = TypeVar("S")
U = TypeVar("U", bound="Tree")


class Tree(Generic[T, U]):
    """A tree data structure
    """

    _data: T
    _children: List[U]
    parent: Optional[U]

    def __init__(self, data: T, children: Iterable[U]) -> None:
        self.data = data
        self.children = list(children)
        self.parent = None

    def get_child(self, idx: int) -> U:
        return self._children[idx]

    def set_child(self, idx: int, tree: U) -> None:
        self._children[idx] = tree
        tree.parent = self

    @property
    def children(self) -> List[U]:
        return self._children

    @children.setter
    def children(self, value: Sequence[U]) -> None:
        self._children = list(value)
        for child in self._children:
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
            return 1 + max([child.depth() for child in self._children])

    def foldl(self, f: Callable[[S, T], S], z: S) -> S:
        res = z
        l = []
        l.append(self)
        while len(l) > 0:
            t = l.pop(0)
            res = f(res, t.data)
            l.extend(t.children)

        return res

    def flatten(self) -> List[T]:
        def append(l, a):
            l.append(a)
            return l

        return self.foldl(append, [])

    def pprint(self, tab: int = 0) -> str:
        return _tree_pprint(self, tab)

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return self.__str__()


def _tree_pprint(tree: Tree, tab: int = 0) -> str:
    pad = " |" * tab + "-"
    if tree is None:
        return pad + "None\n"
    children = [_tree_pprint(child, tab + 1) for child in tree.children]
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
