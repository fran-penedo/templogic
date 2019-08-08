import logging
import unittest

import numpy as np  # type: ignore

from .. import milp_util as milp


class TestSTL(unittest.TestCase):
    def setUp(self) -> None:
        self.m = milp.create_milp("foo")
        self.x0 = self.m.addVar(lb=-milp.GRB.INFINITY, ub=milp.GRB.INFINITY, name="x0")
        self.x1 = self.m.addVar(lb=-milp.GRB.INFINITY, ub=milp.GRB.INFINITY, name="x1")

    def test_max1(self) -> None:
        y = milp.add_max_constr(
            self.m, "max", [self.x0, self.x1], 1000, nnegative=False
        )
        self.m.addConstr(self.x0 == -50.0)
        self.m.addConstr(self.x1 == 50.0)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y["max"].x, 50.0)

    def test_max2(self) -> None:
        y = milp.add_max_constr(
            self.m, "max", [self.x0, self.x1], 1000, nnegative=False
        )
        self.m.addConstr(self.x0 == 50.0)
        self.m.addConstr(self.x1 == -50.0)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y["max"].x, 50.0)

    def test_min1(self) -> None:
        y = milp.add_min_constr(
            self.m, "min", [self.x0, self.x1], 1000, nnegative=False
        )
        self.m.addConstr(self.x0 == -50.0)
        self.m.addConstr(self.x1 == 50.0)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y["min"].x, -50.0)

    def test_min2(self) -> None:
        y = milp.add_min_constr(
            self.m, "min", [self.x0, self.x1], 1000, nnegative=False
        )
        self.m.addConstr(self.x0 == 50.0)
        self.m.addConstr(self.x1 == -50.0)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y["min"].x, -50.0)

    def test_abs(self) -> None:
        y = milp.add_abs_var(self.m, "abs", self.x0, 1)
        r = self.m.addConstr(self.x0 == 50)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y.x, 50)
        self.m.remove(r)
        self.m.addConstr(self.x0 == -50)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y.x, 50)

    def test_set_flag(self) -> None:
        A = np.array([[1, 0], [0, 1]])
        b = np.array([5, 5])
        delta = milp.add_set_flag(self.m, "delta", [self.x0, self.x1], A, b, 1000)
        r1 = self.m.addConstr(self.x0 == 1)
        r2 = self.m.addConstr(self.x1 == 2)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(delta.x, 0)

        self.m.remove(r1)
        self.m.remove(r2)
        r1 = self.m.addConstr(self.x0 == 1)
        r2 = self.m.addConstr(self.x1 == 10)
        self.m.update()
        self.m.optimize()
        # self.m.computeIIS()
        # self.m.write("out.ilp")
        self.assertAlmostEqual(delta.x, 1)

    def test_binary_switch(self) -> None:
        A = np.array([[1, 1]])
        b = np.array([5])
        delta = milp.add_set_flag(self.m, "delta", [self.x0, self.x1], A, b, 1000)
        y = milp.add_binary_switch(self.m, "y", 50, -50, delta, 1000)
        r1 = self.m.addConstr(self.x0 == 1)
        r2 = self.m.addConstr(self.x1 == -2)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y.x, 50)

        self.m.remove(r1)
        self.m.remove(r2)
        r1 = self.m.addConstr(self.x0 == 10)
        r2 = self.m.addConstr(self.x1 == -2)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y.x, -50)

    def test_set_switch(self) -> None:
        A = np.array([[1, 1]])
        b = np.array([5])
        vs = [50, -50]
        y = milp.add_set_switch(
            self.m, "y", [(A, b), (-A, -b)], vs, [self.x0, self.x1], 1000
        )
        r1 = self.m.addConstr(self.x0 == 1)
        r2 = self.m.addConstr(self.x1 == -2)
        self.m.update()
        self.m.optimize()
        self.assertAlmostEqual(y.x, 50)

        self.m.remove(r1)
        self.m.remove(r2)
        r1 = self.m.addConstr(self.x0 == 10)
        r2 = self.m.addConstr(self.x1 == -2)
        self.m.update()
        self.m.optimize()
        # self.m.computeIIS()
        # self.m.write("out.ilp")
        self.assertAlmostEqual(y.x, -50)
