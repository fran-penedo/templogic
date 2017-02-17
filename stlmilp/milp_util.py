import gurobipy as g
from .stl import *

import logging
logger = logging.getLogger('STLMILP')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s %(module)s:%(lineno)d:%(funcName)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logger.setLevel(logging.DEBUG)

GRB = g.GRB

def create_milp(name, minim=True):
    m = g.Model(name)
    m.modelSense = g.GRB.MINIMIZE if minim else g.GRB.MAXIMIZE
    return m

# Common transformations


def delta_label(l, i):
    return l + "_" + str(i)


def add_max_constr(m, label, args, K, nnegative=True):
    return add_minmax_constr(m, label, args, K, 'max', nnegative)


def add_min_constr(m, label, args, K, nnegative=True):
    return add_minmax_constr(m, label, args, K, 'min', nnegative)


# TODO handle len(args) == 2 differently
def add_minmax_constr(m, label, args, K, op='min', nnegative=True):
    """
    Adds the constraint label = op{args} to the milp m

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

    """
    if op not in ['min', 'max']:
        raise ValueError('Expected one of [min, max]')

    y = {}
    y[label] = m.addVar(lb=0 if nnegative else -K, ub=K, name=label)

    if len(args) == 0:
        m.addConstr(y[label] == K)

    else:
        for i in range(len(args)):
            l = delta_label(label, i)
            y[l] = m.addVar(vtype=g.GRB.BINARY, name=l)

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
        m.addConstr(y == expr)
        return y, bounds
    else:
        return None, None


def _stl_not(m, label, f, t):
    x, bounds = add_stl_constr(m, label + "_not", f.args[0], t)
    if x is not None:
        y = m.addVar(name=label, lb=bounds[0], ub=bounds[1])
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
        add = add_min_constr if op == "and" else add_max_constr
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
    for i in range(f.bounds[0], f.bounds[1] + 1):
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
    """
    Adds the stl constraint f at time t to the milp m.

    m : a gurobi Model
    label : a string
            The prefix for the variables added when encoding the constraint
    f : an stl Formula
        The constraint to add. Expressions will be added as the value of the
        signal at the corresponding time using m as the model (i.e., the
        expression variables will be obtained by calling m.getVarByName)
    t : a numeric
        The base time for the constraint

    """
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


# Dynamic System Constraints
def label(name, i, j):
    return name + "_" + str(i) + "_" + str(j)

def unlabel(label):
    sp = label.split("_")
    return sp[0], int(sp[1]), int(sp[2])

def add_affsys_constr_x0(m, l, A, b, x0, N, xhist=None):
    x = add_affsys_constr(m, l, A, b, N, xhist)
    for j in range(1, N):
        for i in [0, A.shape[0] + 1]:
            m.addConstr(x[label(l, i, j)] == x0[i])
    for i in range(A.shape[0] + 2):
        m.addConstr(x[label(l, i, 0)] == x0[i])
    return x



def add_affsys_constr_x0_set(m, l, A, b, N, xpart, pset, f):
    x = add_affsys_constr(m, l, A, b, N, None)

    # p0 = m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name='p0')
    # p1 = m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name='p1')
    p = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('p', i, 0))
          for i in range(pset.shape[1] - 1)]
    m.update()

    for i in range(pset.shape[0]):
        m.addConstr(g.quicksum(
            pset[i][j] * p[j] for j in range(pset.shape[1] - 1)) <= pset[i][-1])
    # for i in range(pset.shape[0]):
    #     m.addConstr(p0 * pset[i][0] + p1 * pset[i][1] <= pset[i][2])

    for i in range(len(xpart)):
        m.addConstr(x[label(l, i, 0)] == f(xpart[i], p))
    for j in range(1, N):
        for i in [0, A.shape[0] + 1]:
            m.addConstr(x[label(l, i, j)] == f(xpart[i], p))

    return x

def add_affsys_constr(m, l, A, b, N, xhist=None):
    if xhist is None:
        xhist = []

    # Decision variables
    logger.debug("Adding decision variables")
    x = {}
    for i in range(A.shape[0] + 2):
        for j in range(-len(xhist), N):
            labelx = label(l, i, j)
            x[labelx] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelx)
    m.update()

    # Dynamics
    logger.debug("Adding dynamics")
    for i in range(1, A.shape[0] + 1):
        logger.debug("Adding row {}".format(i))
        for j in range(-len(xhist), N):
            if j < 0:
                m.addConstr(x[label(l, i, j)] == xhist[len(xhist) + j][i])
            elif j > 0:
                m.addConstr(x[label(l, i, j)] ==
                            g.quicksum(A[i-1][k] * x[label(l, k + 1, j - 1)]
                                       for k in range(A.shape[0])) + b[i-1])
    m.update()

    return x


def add_newmark_constr_x0_set(m, l, M, K, F, xpart, pset, f, dt, N):
    x = add_newmark_constr(m, l, M, K, F, dt, N, None)

    dset, vset, fset = pset
    fd, fv, ff = f

    pd = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pd', i, 0))
          for i in range(dset.shape[1] - 1)]
    pv = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pv', i, 0))
          for i in range(vset.shape[1] - 1)]
    pf = [m.addVar(obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=label('pf', i, 0))
          for i in range(fset.shape[1] - 1)]

    for i in range(dset.shape[0]):
        m.addConstr(g.quicksum(
            pd[j] * dset[i][j] for j in range(dset.shape[1] - 1)) <= dset[i][-1])
    for i in range(vset.shape[0]):
        m.addConstr(g.quicksum(
            pv[j] * vset[i][j] for j in range(vset.shape[1] - 1)) <= vset[i][-1])
    for i in range(fset.shape[0]):
        m.addConstr(g.quicksum(
            pf[j] * fset[i][j] for j in range(fset.shape[1] - 1)) <= fset[i][-1])

    for i in range(len(xpart)):
        m.addConstr(x[label(l, i, 0)] == fd(xpart[i], pd))
        m.addConstr(x[label('d' + l, i, 0)] == fv(xpart[i], pv))
        m.addConstr(x[label('f', i, 0)] == ff(xpart[i], pf))

    return x

def add_newmark_constr_x0(m, l, M, K, F, x0, dt, N, xhist=None):
    x = add_newmark_constr(m, l, M, K, F, dt, N, xhist)
    d0, v0 = x0
    # for j in range(1, N):
    #     for i in [0, M.shape[0] + 1]:
    #         m.addConstr(x[label(l, i, j)] == d0[i])
    #         m.addConstr(x[label('d' + l, i, j)] == v0[i])
    for i in range(M.shape[0]):
        m.addConstr(x[label(l, i, 0)] == d0[i])
        m.addConstr(x[label('d' + l, i, 0)] == v0[i])
    return x

def add_newmark_constr(m, l, M, K, F, dt, N, xhist=None):
    if xhist is None:
        xhist = []

    # Decision variables
    logger.debug("Adding decision variables")
    x = {}
    for i in range(M.shape[0]):
        labelf = label('f', i, 0)
        x[labelf] = m.addVar(
            obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelf)
        for j in range(-len(xhist), N):
            labelx = label(l, i, j)
            x[labelx] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelx)
            labelv = label('d' + l, i, j)
            x[labelv] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labelv)
            labeltv = label('td' + l, i, j)
            x[labeltv] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labeltv)
            labela = label('dd' + l, i, j)
            x[labela] = m.addVar(
                obj=0, lb=-g.GRB.INFINITY, ub=g.GRB.INFINITY, name=labela)

    m.update()

    # Dynamics
    logger.debug("Adding dynamics")
    for i in range(M.shape[0]):
        logger.debug("Adding row {}".format(i))
        for j in range(-len(xhist), N):
            if j < 0:
                m.addConstr(x[label(l, i, j)] == xhist[len(xhist) + j][i])
            elif j == 0:
                # M a = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('dd' + l, i, j)] == 0)
                else:
                    m.addConstr(g.quicksum(M[i, k] * x[label('dd' + l, k, j)]
                                        for k in range(M.shape[0])) ==
                                x[label('f', i, 0)] + F[i] -
                                g.quicksum(K[i, k] * x[label(l, k, j)]
                                                for k in range(K.shape[0])))
            elif j > 0:
                # tv = v + .5 * dt * a
                m.addConstr(x[label('td' + l, i, j)] == x[label('d' + l, i, j - 1)] +
                            .5 * dt * x[label('dd' + l, i, j - 1)])
                # d = d + dt * tv
                m.addConstr(x[label(l, i, j)] == x[label(l, i, j - 1)] +
                            dt * x[label('td' + l, i, j)])
                # M a = F - Kd
                if M[i,i] == 0:
                    m.addConstr(x[label('dd' + l, i, j)] == 0)
                else:
                    m.addConstr(g.quicksum(M[i, k] * x[label('dd' + l, k, j)]
                                        for k in range(M.shape[0])) ==
                                x[label('f', i, 0)] + F[i] -
                                g.quicksum(K[i, k] * x[label(l, k, j)]
                                                for k in range(K.shape[0])))
                # v = tv + .5 * dt * a
                m.addConstr(x[label('d' + l, i, j)] == x[label('td' + l, i, j)] +
                            .5 * dt * x[label('dd' + l, i, j)])
    m.update()

    return x
