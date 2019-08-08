from typing import Sequence, Iterable, Callable, Tuple, Union

import gurobipy as g  # type: ignore
from gurobipy import Model as gModel, Var as gVar

import logging

logger = logging.getLogger(__name__)

GRB = g.GRB


def create_milp(name: str, minim: bool = True) -> gModel:
    m = gModel(name)
    m.modelSense = g.GRB.MINIMIZE if minim else g.GRB.MAXIMIZE
    return m


# Common transformations


def delta_label(l: str, i: int) -> str:
    return l + "_" + str(i)


def add_minmax_constr(
    m: gModel,
    label: str,
    args: Sequence[gVar],
    K: float,
    op: str = "min",
    nnegative: bool = True,
    start: float = GRB.UNDEFINED,
    start_index: int = None,
) -> gVar:
    """Adds the constraint label = op{args} to the milp m

    Parameters
    ----------
    m : a gurobi Model
    label : a string
        Prefix for the variables added
    args : a list of variables
        The set of variables forming the argument of the operation
    K : a numeric
        Must be an upper bound of the absolute value of the variables in args
    op : a value from ['min', 'max']
        The operation to encode
    nnegative : a boolean
        True if 0 is lower bound of all variables in args
    start : float
        Start optimal value for the minmax variable
    start_index : int
        Index of the start optimal value in args

    TODO handle len(args) == 2 differently
    """
    if op not in ["min", "max"]:
        raise ValueError("Expected one of [min, max]")

    y = {}
    y[label] = m.addVar(lb=0 if nnegative else -K, ub=K, name=label)
    y[label].start = start

    if len(args) == 0:
        m.addConstr(y[label] == K)

    else:
        for i in range(len(args)):
            l = delta_label(label, i)
            if start_index is not None:
                start_delta = 1 if start_index == i else 0
            else:
                start_delta = GRB.UNDEFINED
            y[l] = m.addVar(vtype=g.GRB.BINARY, name=l)
            y[l].start = start_delta

        for i in range(len(args)):
            x = delta_label(label, i)
            if op == "min":
                m.addConstr(y[label] <= args[i])
                m.addConstr(y[label] >= args[i] - (1 - y[x]) * K)
            else:
                m.addConstr(y[label] >= args[i])
                m.addConstr(y[label] <= args[i] + (1 - y[x]) * K)

        m.addConstr(
            g.quicksum([y[delta_label(label, i)] for i in range(len(args))]) == 1
        )

    return y


def add_max_constr(
    m: gModel,
    label: str,
    args: Sequence[gVar],
    K: float,
    nnegative: bool = True,
    start: float = GRB.UNDEFINED,
    start_index: int = None,
) -> gVar:
    return add_minmax_constr(m, label, args, K, "max", nnegative, start, start_index)


def add_min_constr(
    m: gModel,
    label: str,
    args: Sequence[gVar],
    K: float,
    nnegative: bool = True,
    start: float = GRB.UNDEFINED,
    start_index: int = None,
) -> gVar:
    return add_minmax_constr(m, label, args, K, "min", nnegative, start, start_index)


def add_abs_var(m: gModel, label: str, var: gVar, obj: float) -> gVar:
    if m.modelSense == g.GRB.MINIMIZE:
        z = m.addVar(name="abs_" + label, obj=obj)
        m.update()
        m.addConstr(z >= var)
        m.addConstr(z >= -var)
        return z
    else:
        raise NotImplementedError()


def add_penalty(m: gModel, label: str, var: gVar, obj: float) -> gVar:
    m.setAttr("OBJ", [var], [-obj])
    y = add_abs_var(m, label, var, obj)
    return y


def add_set_flag(
    m: gModel,
    label: str,
    x_var: gVar,
    A: Sequence[Sequence[float]],
    b: Sequence[float],
    K: float,
) -> gVar:
    if len(A) == 1:
        delta = m.addVar(vtype=g.GRB.BINARY, name=label)
        deltas = [delta]
    else:
        deltas = [
            m.addVar(vtype=g.GRB.BINARY, name=delta_label(label, i))
            for i in range(len(A))
        ]
        delta = m.addVar(vtype=g.GRB.BINARY, name=label)

    for i in range(len(A)):
        m.addConstr(
            g.quicksum(A[i][j] * x_var[j] for j in range(len(x_var))) - deltas[i] * K
            <= b[i]
        )
        m.addConstr(
            g.quicksum(A[i][j] * x_var[j] for j in range(len(x_var)))
            + (1 - deltas[i]) * K
            >= b[i]
        )
    if len(A) > 1:
        for i in range(len(A)):
            m.addConstr(delta >= deltas[i])
        m.addConstr(delta <= g.quicksum(deltas))

    return delta


def add_binary_switch(
    m: gModel, label: str, v1: float, v2: float, delta: gVar, K: float
) -> gVar:
    y = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=label)

    m.addConstr(y <= v1 + delta * K)
    m.addConstr(y >= v1 - delta * K)

    m.addConstr(y <= v2 + (1 - delta) * K)
    m.addConstr(y >= v2 - (1 - delta) * K)

    return y


def add_set_switch(
    m: gModel,
    label: str,
    sets: Sequence[Tuple],
    vs: Sequence[float],
    x_var: gVar,
    K: float,
) -> gVar:
    if len(sets) != 2:
        raise NotImplementedError("Only implemented for two complementary sets")

    delta = add_set_flag(m, label + "_delta", x_var, sets[0][0], sets[0][1], K)
    y = add_binary_switch(m, label, vs[0], vs[1], delta, K)
    return y
