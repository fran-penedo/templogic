import logging

import stl
from milp_util import add_min_constr, add_max_constr, add_penalty, create_milp, GRB

logger = logging.getLogger(__name__)

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
    if f.args[0].op == stl.NOT:
        return add_stl_constr(m, label, f.args[0].args[0], t)
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
    # if f.bounds[0] == f.bounds[1]:
    #     b1 = f.bounds[0]
    #     b2 = f.bounds[1] + 1
    # else:
    b1, b2 = f.bounds
    for i in range(b1, b2 + 1):
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


def build_and_solve(
    spec, model_encode_f, spec_obj,
    outputflag=None, numericfocus=None, threads=4, log_files=True):
    # print spec
    if spec is not None:
        hd = max(0, spec.horizon()) + 1
    else:
        hd = 0

    m = create_milp("rhc_system")
    logger.debug("Adding system constraints")
    model_encode_f(m, hd)
    # sys_milp.add_sys_constr_x0(m, "d", system, d0, hd, None)
    if spec is not None:
        logger.debug("Adding STL constraints")
        fvar, vbds = add_stl_constr(m, "spec", spec)
        fvar.setAttr("obj", spec_obj)

    if outputflag is not None:
        # 0
        m.params.outputflag = outputflag
    if numericfocus is not None:
        # 3
        m.params.numericfocus = numericfocus
    if threads is not None:
        # 4
        m.params.threads = threads
    m.update()
    if log_files:
        m.write("out.lp")
    logger.debug(
        "Optimizing MILP with {} variables ({} binary) and {} constraints".format(
            m.numvars, m.numbinvars, m.numconstrs))
    m.optimize()
    logger.debug("Finished optimizing")
    if log_files:
        f = open("out_vars.txt", "w")
        for v in m.getVars():
            print >> f, v
        f.close()

    if m.status != GRB.status.OPTIMAL:
        logger.warning("MILP returned status: {}".format(m.status))
    return m

