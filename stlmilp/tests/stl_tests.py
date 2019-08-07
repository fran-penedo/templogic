import unittest
from types import MethodType
from typing import Sequence, Callable

import stlmilp.stl as stl

import logging

logger = logging.getLogger(__name__)


class _Model(stl.STLModel):
    def __init__(self, s) -> None:
        self.s = s
        self.tinter = 1

    def getVarByName(self, j):
        return self.s[j]


class TestSTL(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [lambda x: x]
        self.signal: stl.Signal[float] = stl.Signal(self.labels, lambda x: x[0] + 3)
        self.f = stl.STLNext(
            stl.STLAlways(
                bounds=(3, 5),
                arg=stl.STLEventually(
                    bounds=(2, 4),
                    arg=stl.STLAnd(
                        args=[
                            stl.STLOr(
                                args=[
                                    stl.STLPred(self.signal),
                                    stl.STLAlways(
                                        bounds=(1, 3), arg=stl.STLPred(self.signal)
                                    ),
                                    stl.STLEventually(
                                        bounds=(0, 2), arg=stl.STLPred(self.signal)
                                    ),
                                ]
                            ),
                            stl.STLNot(stl.STLPred(self.signal)),
                        ]
                    ),
                ),
            )
        )
        self.s = [0, 1, 2, 4, 8, 4, 2, 1, 0, 1, 2, 6, 2, 1, 5, 7, 8, 1]

        self.model = _Model(self.s)

    def test_horizon(self) -> None:
        self.assertEqual(self.f.horizon(), 13)

    def test_robustness_expr(self) -> None:
        f = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 3))
        self.assertEqual(f.robustness(self.model, 2), self.s[2] + 3)

    def test_robustness_not(self) -> None:
        g = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 3))
        f = stl.STLNot(g)
        self.assertEqual(f.robustness(self.model, 2), -(self.s[2] + 3))

    def test_robustness_and(self) -> None:
        g = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 3))
        h = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 1))
        f = stl.STLAnd([g, h])
        self.assertEqual(f.robustness(self.model, 2), min(self.s[2] + 3, self.s[2] + 1))

    def test_robustness_or(self) -> None:
        g = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 3))
        h = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 1))
        f = stl.STLOr([g, h])
        self.assertEqual(f.robustness(self.model, 2), max(self.s[2] + 3, self.s[2] + 1))

    def test_robustness_next(self) -> None:
        g = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 3))
        f = stl.STLNext(g)
        self.assertEqual(f.robustness(self.model, 2), self.s[3] + 3)

    def test_robustness_always(self) -> None:
        g = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 3))
        f = stl.STLAlways((1, 3), g)
        self.assertEqual(
            f.robustness(self.model, 2),
            min(self.s[3] + 3, self.s[4] + 3, self.s[5] + 3),
        )

    def test_robustness_eventually(self) -> None:
        g = stl.STLPred(stl.Signal(self.labels, lambda x: x[0] + 3))
        f = stl.STLEventually((1, 3), g)
        self.assertEqual(
            f.robustness(self.model, 2),
            max(self.s[3] + 3, self.s[4] + 3, self.s[5] + 3),
        )

    def test_robustness(self) -> None:
        self.assertEqual(self.f.robustness(self.model, 0), -3)
        self.signal.labels = lambda x: [x]
        self.assertEqual(self.f.robustness(self.model, 0), -3)

    def test_robustness_tree(self) -> None:
        tree = stl.robustness_tree(self.f, self.model, 0)
        self.assertEqual(tree.robustness, -3)
        print(tree.pprint())

    def test_parser(self) -> None:
        parser = stl.stl_parser()

        f = stl.STLNext(stl.STLPred(self.signal))
        logger.debug(str(f))
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

    def test_perturb(self) -> None:
        eps = lambda x: x
        setattr(eps, "perturbed", 0)

        def perturb(self, eps) -> "stl.Signal":
            eps.perturbed += 1
            return self

        setattr(self.signal, "perturb", MethodType(perturb, self.signal))

        stl.perturb(self.f, eps)
        self.assertEqual(eps.perturbed, 4)  # type: ignore
        f = stl.STLNot(self.f)
        with self.assertRaises(Exception):
            stl.perturb(f, eps)

    def test_scale_time(self) -> None:
        forig = str(self.f)
        stl.scale_time(self.f, 0.5)
        self.assertEqual(self.f.args[0].bounds[0], 6)
        stl.scale_time(self.f, 2.0)
        self.assertEqual(str(self.f), forig)
