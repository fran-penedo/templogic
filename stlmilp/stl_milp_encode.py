import stl
from milp_util import add_min_constr, add_max_constr, add_penalty

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
    for i in range(f.bounds[0], f.bounds[1]):
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
        stl.EXPR: _stl_expr,
        stl.NOT: _stl_not,
        stl.AND: _stl_and,
        stl.OR: _stl_or,
        stl.ALWAYS: _stl_always,
        stl.NEXT: _stl_next,
        stl.EVENTUALLY: _stl_eventually
    }[f.op](m, label, f, t)

def add_always_constr(m, label, a, b, rho, K, t=0):
    y = add_min_constr(m, label, rho[(t + a):(t + b + 1)], K)[label]
    return y

def add_always_penalized(m, label, a, b, rho, K, obj, t=0):
    y = add_always_constr(m, label, a, b, rho, K, t)
    add_penalty(m, label, y, obj)
    return y

