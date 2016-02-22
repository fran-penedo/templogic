import gurobipy as g
from stl import *


# Common transformations


def delta_label(l, i):
    return l + "_" + str(i)


def add_max_constr(m, label, args, K, nnegative=True):
    return add_minmax_constr(m, label, args, K, 'max', nnegative)


def add_min_constr(m, label, args, K, nnegative=True):
    return add_minmax_constr(m, label, args, K, 'min', nnegative)


# TODO handle len(args) == 2 differently
def add_minmax_constr(m, label, args, K, op='min', nnegative=True):
    if op not in ['min', 'max']:
        raise ValueError('Expected one of [min, max]')

    y = {}
    y[label] = m.addVar(lb=0 if nnegative else -K, ub=K, name=label)

    if len(args) == 0:
        m.update()
        m.addConstr(y[label] == K)

    else:
        for i in range(len(args)):
            l = delta_label(label, i)
            y[l] = m.addVar(vtype=g.GRB.BINARY, name=l)

        m.update()

        for i in range(len(args)):
            x = delta_label(label, i)
            if op == 'min':
                m.addConstr(y[label] <= args[i])
                m.addConstr(y[label] >= args[i] - y[x] * K)
            else:
                m.addConstr(y[label] >= args[i])
                m.addConstr(y[label] <= args[i] + y[x] * K)

        m.addConstr(g.quicksum([y[delta_label(label, i)]
                                for i in range(len(args))]) == len(args) - 1)

    return y


def add_abs_var(m, label, var, obj):
    if m.modelSense == g.GRB.MINIMIZE:
        z = m.addVar(name="abs_"+label, obj=obj)
        m.update()
        m.addConstr(z >= var)
        m.addConstr(z >= -var)
        return z

    else:
        return None


# MILP transformation

def add_milp_var(m, label, delta, x, M, mm):
    y = m.addVar(name=label, lb=min(mm, 0), ub=g.GRB.INFINITY)
    m.update()
    m.addConstr(y <= M * delta)
    m.addConstr(y >= mm * delta)
    m.addConstr(y <= x - mm * (1 - delta))
    m.addConstr(y >= x - M * (1 - delta))
    return y


# STL related transformations

def _stl_expr(m, label, f, t):
    expr = f.args[0].signal(m, t)
    if expr is not None:
        bounds = f.args[0].bounds
        y = m.addVar(name=label, lb=bounds[0], ub=bounds[1])
        m.update()
        m.addConstr(y == expr)
        return y, bounds
    else:
        return None, None


def _stl_not(m, label, f, t):
    x, bounds = add_stl_constr(m, label + "_not", f.args[0], t)
    if x is not None:
        y = m.addVar(name=label, lb=bounds[0], ub=bounds[1])
        m.update()
        m.addConstr(y == -x)
        return y, bounds
    else:
        return None, None


def _stl_and_or(m, label, f, t, op):
    xx = []
    boundss = []
    for i, ff in enumerate(f.args):
        x, bounds = add_stl_constr(m, label + "_" + op + str(i), ff, t)
        if x is not None:
            xx.append(x)
            boundss.append(bounds)

    if len(xx) > 0:
        # I'm not gonna bother using the best bounds
        bounds = map(max, zip(*boundss))
        K = max(map(abs, bounds))
        add = add_min_constr if op == "min" else add_max_constr
        y = add(m, label, xx, K, nnegative=False)[label]
        return y, bounds

    else:
        return None, None


def _stl_and(m, label, f, t):
    return _stl_and_or(m, label, f, t, "and")


def _stl_or(m, label, f, t):
    return _stl_and_or(m, label, f, t, "or")


def _stl_next(m, label, f, t):
    return add_stl_constr(m, label, f.args[0], t+1)


def _stl_always_eventually(m, label, f, t, op):
    xx = []
    boundss = []
    for i in range(f.bnd[0], f.bnd[1] + 1):
        x, bounds = add_stl_constr(m, label + "_" + op + str(i), f.args[0],
                                   t + i)
        if x is not None:
            xx.append(x)
            boundss.append(bounds)

    if len(xx) > 0:
        # I'm not gonna bother using the best bounds
        bounds = map(max, zip(*boundss))
        K = max(map(abs, bounds))
        add = add_min_constr if op == "alw" else add_max_constr
        y = add(m, label, xx, K, nnegative=False)[label]
        return y, bounds

    else:
        return None, None


def _stl_always(m, label, f, t):
    return _stl_always_eventually(m, label, f, t, "alw")


def _stl_eventually(m, label, f, t):
    return _stl_always_eventually(m, label, f, t, "eve")


def add_stl_constr(m, label, f, t=0):
    return {
        EXPR: _stl_expr,
        NOT: _stl_not,
        AND: _stl_and,
        OR: _stl_or,
        ALWAYS: _stl_always,
        NEXT: _stl_next,
        EVENTUALLY: _stl_eventually
    }[f.op](m, label, f, t)


def add_penalty(m, label, var, obj):
    m.setAttr("OBJ", [var], [-obj])
    y = add_abs_var(m, label, var, obj)
    return y


def add_always_constr(m, label, a, b, rho, K, t=0):
    y = add_min_constr(m, label, rho[(t + a):(t + b + 1)], K)[label]
    return y


def add_always_penalized(m, label, a, b, rho, K, obj, t=0):
    y = add_always_constr(m, label, a, b, rho, K, t)
    add_penalty(m, label, y, obj)
    return y

