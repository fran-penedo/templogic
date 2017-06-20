import logging
import unittest

import stlmilp.stl as stl

class TestSTL(unittest.TestCase):
    def setUp(self):
        self.f = stl.Formula(stl.NEXT, [
            stl.Formula(stl.ALWAYS, bounds=[3, 5], args=[
                stl.Formula(stl.EVENTUALLY, bounds=[2, 4], args=[
                    stl.Formula(stl.AND, args=[
                        stl.Formula(stl.OR, args=[
                            stl.Formula(stl.EXPR, []),
                            stl.Formula(stl.ALWAYS, bounds=[1, 3],
                                    args=[stl.Formula(stl.EXPR, [])]),
                            stl.Formula(stl.EVENTUALLY, bounds=[0, 2],
                                    args=[stl.Formula(stl.EXPR, [])])
                        ]),
                        stl.Formula(stl.NOT, [stl.Formula(stl.EXPR, [])])
                    ])
                ])
            ])
        ])
        self.s = [0, 1, 2, 4, 8, 4, 2, 1, 0]
        class Model():
            def __init__(self, s):
                self.s = s
                self.tinter = 1

            def getVarByName(self, j):
                return self.s[j]

        self.model = Model(self.s)
        self.labels = [lambda x: x]

    def test_horizon(self):
        assert self.f.horizon() == 13

    def test_robustness_expr(self):
        f = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        assert(stl.robustness(f, self.model, 2) == self.s[2] + 3)

    def test_robustness_not(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.NOT, [g])
        assert(-stl.robustness(f, self.model, 2) == self.s[2] + 3)

    def test_robustness_and(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        h = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 1)])
        f = stl.Formula(stl.AND, [g, h])
        assert(stl.robustness(f, self.model, 2) == min(self.s[2] + 3, self.s[2] + 1))

    def test_robustness_or(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        h = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 1)])
        f = stl.Formula(stl.OR, [g, h])
        assert(stl.robustness(f, self.model, 2) == max(self.s[2] + 3, self.s[2] + 1))

    def test_robustness_next(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.NEXT, [g])
        assert(stl.robustness(f, self.model, 2) == self.s[3] + 3)

    def test_robustness_always(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.ALWAYS, [g], [1, 3])
        assert(stl.robustness(f, self.model, 2) == min(self.s[3] + 3, self.s[4] + 3, self.s[5] + 3))

    def test_robustness_eventually(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.EVENTUALLY, [g], [1, 3])
        assert(stl.robustness(f, self.model, 2) == max(self.s[3] + 3, self.s[4] + 3, self.s[5] + 3))
