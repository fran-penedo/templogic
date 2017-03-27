import gurobipy as g

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

    TODO handle len(args) == 2 differently
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

def add_max_constr(m, label, args, K, nnegative=True):
    return add_minmax_constr(m, label, args, K, 'max', nnegative)

def add_min_constr(m, label, args, K, nnegative=True):
    return add_minmax_constr(m, label, args, K, 'min', nnegative)

def add_abs_var(m, label, var, obj):
    if m.modelSense == g.GRB.MINIMIZE:
        z = m.addVar(name="abs_"+label, obj=obj)
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
