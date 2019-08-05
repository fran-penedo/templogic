import gurobipy as g

import logging

logger = logging.getLogger(__name__)

GRB = g.GRB


def create_milp(name, minim=True):
    m = g.Model(name)
    m.modelSense = g.GRB.MINIMIZE if minim else g.GRB.MAXIMIZE
    return m


# Common transformations


def delta_label(l, i):
    return l + "_" + str(i)


def add_minmax_constr(
    m, label, args, K, op="min", nnegative=True, start=GRB.UNDEFINED, start_index=None
):
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
    m, label, args, K, nnegative=True, start=GRB.UNDEFINED, start_index=None
):
    return add_minmax_constr(m, label, args, K, "max", nnegative, start, start_index)


def add_min_constr(
    m, label, args, K, nnegative=True, start=GRB.UNDEFINED, start_index=None
):
    return add_minmax_constr(m, label, args, K, "min", nnegative, start, start_index)


def add_abs_var(m, label, var, obj):
    if m.modelSense == g.GRB.MINIMIZE:
        z = m.addVar(name="abs_" + label, obj=obj)
        m.update()
        m.addConstr(z >= var)
        m.addConstr(z >= -var)
        return z

    else:
        return None


def add_penalty(m, label, var, obj):
    m.setAttr("OBJ", [var], [-obj])
    y = add_abs_var(m, label, var, obj)
    return y


def add_set_flag(m, label, x_var, A, b, K):
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


def add_binary_switch(m, label, v1, v2, delta, K):
    y = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=label)

    m.addConstr(y <= v1 + delta * K)
    m.addConstr(y >= v1 - delta * K)

    m.addConstr(y <= v2 + (1 - delta) * K)
    m.addConstr(y >= v2 - (1 - delta) * K)

    return y


def add_set_switch(m, label, sets, vs, x_var, K):
    if len(sets) != 2:
        raise NotImplementedError("Only implemented for two complementary sets")

    delta = add_set_flag(m, label + "_delta", x_var, sets[0][0], sets[0][1], K)
    y = add_binary_switch(m, label, vs[0], vs[1], delta, K)
    return y
