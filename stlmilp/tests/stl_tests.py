import unittest

import stlmilp.stl as stl

import logging

logger = logging.getLogger(__name__)


class _Model:
    def __init__(self, s):
        self.s = s
        self.tinter = 1

    def getVarByName(self, j):
        return self.s[j]


class TestSTL(unittest.TestCase):
    def setUp(self):
        self.labels = [lambda x: x]
        self.signal = stl.Signal(self.labels, lambda x: x[0] + 3)
        self.f = stl.Formula(
            stl.NEXT,
            [
                stl.Formula(
                    stl.ALWAYS,
                    bounds=[3, 5],
                    args=[
                        stl.Formula(
                            stl.EVENTUALLY,
                            bounds=[2, 4],
                            args=[
                                stl.Formula(
                                    stl.AND,
                                    args=[
                                        stl.Formula(
                                            stl.OR,
                                            args=[
                                                stl.Formula(stl.EXPR, [self.signal]),
                                                stl.Formula(
                                                    stl.ALWAYS,
                                                    bounds=[1, 3],
                                                    args=[
                                                        stl.Formula(
                                                            stl.EXPR, [self.signal]
                                                        )
                                                    ],
                                                ),
                                                stl.Formula(
                                                    stl.EVENTUALLY,
                                                    bounds=[0, 2],
                                                    args=[
                                                        stl.Formula(
                                                            stl.EXPR, [self.signal]
                                                        )
                                                    ],
                                                ),
                                            ],
                                        ),
                                        stl.Formula(
                                            stl.NOT,
                                            [stl.Formula(stl.EXPR, [self.signal])],
                                        ),
                                    ],
                                )
                            ],
                        )
                    ],
                )
            ],
        )
        self.s = [0, 1, 2, 4, 8, 4, 2, 1, 0, 1, 2, 6, 2, 1, 5, 7, 8, 1]

        self.model = _Model(self.s)

    def test_horizon(self):
        assert self.f.horizon() == 13

    def test_robustness_expr(self):
        f = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        assert stl.robustness(f, self.model, 2) == self.s[2] + 3

    def test_robustness_not(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.NOT, [g])
        assert -stl.robustness(f, self.model, 2) == self.s[2] + 3

    def test_robustness_and(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        h = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 1)])
        f = stl.Formula(stl.AND, [g, h])
        assert stl.robustness(f, self.model, 2) == min(self.s[2] + 3, self.s[2] + 1)

    def test_robustness_or(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        h = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 1)])
        f = stl.Formula(stl.OR, [g, h])
        assert stl.robustness(f, self.model, 2) == max(self.s[2] + 3, self.s[2] + 1)

    def test_robustness_next(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.NEXT, [g])
        assert stl.robustness(f, self.model, 2) == self.s[3] + 3

    def test_robustness_always(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.ALWAYS, [g], [1, 3])
        assert stl.robustness(f, self.model, 2) == min(
            self.s[3] + 3, self.s[4] + 3, self.s[5] + 3
        )

    def test_robustness_eventually(self):
        g = stl.Formula(stl.EXPR, [stl.Signal(self.labels, lambda x: x[0] + 3)])
        f = stl.Formula(stl.EVENTUALLY, [g], [1, 3])
        assert stl.robustness(f, self.model, 2) == max(
            self.s[3] + 3, self.s[4] + 3, self.s[5] + 3
        )

    def test_robustness(self):
        self.assertEqual(stl.robustness(self.f, self.model, 0), -3)
        self.signal.labels = lambda x: [x]
        self.assertEqual(stl.robustness(self.f, self.model, 0), -3)

    def test_robustness_tree(self):
        tree = stl.robustness_tree(self.f, self.model, 0)
        self.assertEqual(tree.robustness, -3)
        print(tree.pprint())

    def test_parser(self):
        parser = stl.stl_parser()

        f = stl.Formula(stl.NEXT, [stl.Formula(stl.EXPR, [self.signal])])
        form = parser.parseString(str(f))[0]
        self.assertEqual(str(f), str(form))

        fstr = "O G_[3.00, 5.00] F_[2.00, 4.00] (EXP)"
        form = parser.parseString(fstr)[0]
        self.assertEqual(fstr, str(form))

        fstr = "~ (EXP)"
        form = parser.parseString(fstr)[0]
        self.assertEqual(fstr, str(form))

        fstr = "((EXP) | ~ (EXP))"
        form = parser.parseString(fstr)[0]
        self.assertEqual(fstr, str(form))

        fstr = "O G_[3.00, 5.00] F_[2.00, 4.00] ((EXP) & ~ (EXP))"
        form = parser.parseString(fstr)[0]
        self.assertEqual(fstr, str(form))

        fstr = "O G_[3.00, 5.00] F_[2.00, 4.00] (((EXP) | G_[1.00, 3.00] (EXP) | F_[0.00, 2.00] (EXP)) | ~ (EXP))"
        form = parser.parseString(fstr)[0]
        self.assertEqual(fstr, str(form))

        fstr = str(self.f)
        form = parser.parseString(fstr)[0]
        self.assertEqual(fstr, str(form))

    def test_perturb(self):
        eps = lambda: None
        setattr(eps, "perturbed", 0)

        def perturb(eps):
            eps.perturbed += 1

        self.signal.perturb = perturb
        stl.perturb(self.f, eps)
        self.assertEqual(eps.perturbed, 4)
        f = stl.Formula(stl.NOT, [self.f])
        with self.assertRaises(Exception):
            stl.perturb(f, eps)

    def test_scale_time(self):
        forig = str(self.f)
        stl.scale_time(self.f, 0.5)
        self.assertEqual(self.f.args[0].bounds[0], 6)
        stl.scale_time(self.f, 2.0)
        self.assertEqual(str(self.f), forig)
