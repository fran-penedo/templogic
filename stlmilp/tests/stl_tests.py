from ..stl import *


def test_horizon():
    f = Formula(NEXT, [
        Formula(ALWAYS, bounds=[3, 5], args=[
            Formula(EVENTUALLY, bounds=[2, 4], args=[
                Formula(AND, args=[
                    Formula(OR, args=[
                        Formula(EXPR, []),
                        Formula(ALWAYS, bounds=[1, 3],
                                args=[Formula(EXPR, [])]),
                        Formula(EVENTUALLY, bounds=[0, 2],
                                args=[Formula(EXPR, [])])
                    ]),
                    Formula(NOT, [Formula(EXPR, [])])
                ])
            ])
        ])
    ])

    assert f.horizon() == 13


s = [0, 1, 2, 4, 8, 4, 2, 1, 0]
class Model():
    def __init__(self, s):
        self.s = s
        self.tinter = 1

    def getVarByName(self, j):
        return self.s[j]

model = Model(s)
labels = [lambda x: x]


def test_robustness_expr():
    f = Formula(EXPR, [Signal(labels, lambda x: x[0] + 3)])
    assert(robustness(f, model, 2) == s[2] + 3)


def test_robustness_not():
    g = Formula(EXPR, [Signal(labels, lambda x: x[0] + 3)])
    f = Formula(NOT, [g])
    assert(-robustness(f, model, 2) == s[2] + 3)


def test_robustness_and():
    g = Formula(EXPR, [Signal(labels, lambda x: x[0] + 3)])
    h = Formula(EXPR, [Signal(labels, lambda x: x[0] + 1)])
    f = Formula(AND, [g, h])
    assert(robustness(f, model, 2) == min(s[2] + 3, s[2] + 1))


def test_robustness_or():
    g = Formula(EXPR, [Signal(labels, lambda x: x[0] + 3)])
    h = Formula(EXPR, [Signal(labels, lambda x: x[0] + 1)])
    f = Formula(OR, [g, h])
    assert(robustness(f, model, 2) == max(s[2] + 3, s[2] + 1))


def test_robustness_next():
    g = Formula(EXPR, [Signal(labels, lambda x: x[0] + 3)])
    f = Formula(NEXT, [g])
    assert(robustness(f, model, 2) == s[3] + 3)


def test_robustness_always():
    g = Formula(EXPR, [Signal(labels, lambda x: x[0] + 3)])
    f = Formula(ALWAYS, [g], [1, 3])
    assert(robustness(f, model, 2) == min(s[3] + 3, s[4] + 3, s[5] + 3))


def test_robustness_eventually():
    g = Formula(EXPR, [Signal(labels, lambda x: x[0] + 3)])
    f = Formula(EVENTUALLY, [g], [1, 3])
    assert(robustness(f, model, 2) == max(s[3] + 3, s[4] + 3, s[5] + 3))
