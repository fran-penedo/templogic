from typing import (
    Sequence,
    Iterable,
    Callable,
    Tuple,
    Union,
    cast,
    Optional,
    Mapping,
    Type,
)
import logging

from . import stl
from .milp_util import (
    add_min_constr,
    add_max_constr,
    add_penalty,
    create_milp,
    GRB,
    gModel,
    gVar,
)

logger = logging.getLogger(__name__)


def _stl_expr(
    m: gModel,
    label: str,
    f: stl.STLPred,
    t: float,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    expr = f.signal.signal(m, t)
    if expr is not None:
        bounds = f.signal.bounds
        if start_robustness_tree is not None:
            r = start_robustness_tree.robustness
        else:
            r = GRB.UNDEFINED
        y = m.addVar(name=label, lb=bounds[0], ub=bounds[1])
        y.start = r
        m.addConstr(y == expr)
        return y, bounds
    else:
        raise NotImplementedError()


def _stl_not(
    m: gModel,
    label: str,
    f: stl.STLNot,
    t: float,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    if start_robustness_tree is not None:
        tree: Optional[stl.RobustnessTree] = start_robustness_tree.children[0]
    else:
        tree = None
    if isinstance(f.args[0], stl.STLNot):
        if tree is not None:
            tree = tree.children[0]
        return add_stl_constr(m, label, f.args[0].args[0], t, tree)
    x, bounds = add_stl_constr(m, label + "_not", f.args[0], t, tree)
    if x is not None:
        if start_robustness_tree is not None:
            r = start_robustness_tree.robustness
        else:
            r = GRB.UNDEFINED
        y = m.addVar(name=label, lb=bounds[0], ub=bounds[1])
        y.start = r
        m.addConstr(y == -x)
        return y, bounds
    else:
        raise NotImplementedError()


def _stl_and_or(
    m: gModel,
    label: str,
    f: Union[stl.STLAnd, stl.STLOr],
    t: float,
    op: str,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    xx = []
    boundss = []
    for i, ff in enumerate(f.args):
        if start_robustness_tree is not None:
            tree: Optional[stl.RobustnessTree] = start_robustness_tree.children[i]
        else:
            tree = None
        x, bounds = add_stl_constr(m, label + "_" + op + str(i), ff, t, tree)
        if x is not None:
            xx.append(x)
            boundss.append(bounds)

    if len(xx) > 0:
        # I'm not gonna bother using the best bounds
        bounds = list(map(max, zip(*boundss)))  # type: ignore # zip issues
        K = max([abs(b) for b in bounds])
        add = add_min_constr if op == "and" else add_max_constr
        if start_robustness_tree is not None:
            r, index = start_robustness_tree.robustness, start_robustness_tree.index
        else:
            r, index = GRB.UNDEFINED, None  # type: ignore
        y = add(m, label, xx, K, nnegative=False, start=r, start_index=index)[label]
        return y, bounds

    else:
        raise NotImplementedError()


def _stl_and(
    m: gModel,
    label: str,
    f: stl.STLAnd,
    t: float,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    return _stl_and_or(m, label, f, t, "and", start_robustness_tree)


def _stl_or(
    m: gModel,
    label: str,
    f: stl.STLOr,
    t: float,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    return _stl_and_or(m, label, f, t, "or", start_robustness_tree)


def _stl_next(
    m: gModel,
    label: str,
    f: stl.STLNext,
    t: float,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    return add_stl_constr(m, label, f.args[0], t + 1, start_robustness_tree)


def _stl_always_eventually(
    m: gModel,
    label: str,
    f: Union[stl.STLEventually, stl.STLAlways],
    t: float,
    op: str,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    xx = []
    boundss = []
    # if f.bounds[0] == f.bounds[1]:
    #     b1 = f.bounds[0]
    #     b2 = f.bounds[1] + 1
    # else:
    b1, b2 = [int(b) for b in f.bounds]
    for i in range(b1, b2 + 1):
        if start_robustness_tree is not None:
            tree = start_robustness_tree.children[i - b1]
        else:
            tree = None
        x, bounds = add_stl_constr(m, label + "_" + op + str(i), f.args[0], t + i, tree)
        if x is not None:
            xx.append(x)
            boundss.append(bounds)

    if len(xx) > 0:
        # I'm not gonna bother using the best bounds
        bounds = list(map(max, zip(*boundss)))  # type: ignore
        K = max([abs(b) for b in bounds])
        add = add_min_constr if op == "alw" else add_max_constr
        if start_robustness_tree is not None:
            r, index = start_robustness_tree.robustness, start_robustness_tree.index
        else:
            r, index = GRB.UNDEFINED, None  # type: ignore
        y = add(m, label, xx, K, nnegative=False, start=r, start_index=index)[label]
        return y, bounds

    else:
        raise NotImplementedError()


def _stl_always(
    m: gModel,
    label: str,
    f: stl.STLAlways,
    t: float,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    return _stl_always_eventually(m, label, f, t, "alw", start_robustness_tree)


def _stl_eventually(
    m: gModel,
    label: str,
    f: stl.STLEventually,
    t: float,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    return _stl_always_eventually(m, label, f, t, "eve", start_robustness_tree)


def add_stl_constr(
    m: gModel,
    label: str,
    f: stl.STLTerm,
    t: float = 0,
    start_robustness_tree: stl.RobustnessTree = None,
) -> Tuple[gVar, Tuple[float, float]]:
    """Adds the stl constraint f at time t to the milp m.

    Parameters
    ----------
    m : a gurobi Model
    label : a string
        The prefix for the variables added when encoding the constraint
    f : an stl Formula
        The constraint to add. Expressions will be added as the value of the
        signal at the corresponding time using m as the model (i.e., the
        expression variables will be obtained by calling m.getVarByName)
    t : a numeric
        The base time for the constraint
    start_robustness_tree : RobustnessTree
        Use information from this robustness tree to set the start vector of
        the MIP

    """

    if isinstance(f, stl.STLPred):
        return _stl_expr(m, label, f, t, start_robustness_tree)
    elif isinstance(f, stl.STLNot):
        return _stl_not(m, label, f, t, start_robustness_tree)
    elif isinstance(f, stl.STLAnd):
        return _stl_and(m, label, f, t, start_robustness_tree)
    elif isinstance(f, stl.STLOr):
        return _stl_or(m, label, f, t, start_robustness_tree)
    elif isinstance(f, stl.STLAlways):
        return _stl_always(m, label, f, t, start_robustness_tree)
    elif isinstance(f, stl.STLNext):
        return _stl_next(m, label, f, t, start_robustness_tree)
    elif isinstance(f, stl.STLEventually):
        return _stl_eventually(m, label, f, t, start_robustness_tree)
    else:
        raise Exception("Non exhaustive pattern matching")


def add_always_constr(
    m: gModel, label: str, a: int, b: int, rho: Sequence[gVar], K: float, t: int = 0
) -> gVar:
    y = add_min_constr(m, label, rho[(t + a) : (t + b + 1)], K)[label]
    return y


def add_always_penalized(
    m: gModel,
    label: str,
    a: int,
    b: int,
    rho: Sequence[gVar],
    K: float,
    obj: float,
    t: int = 0,
) -> gVar:
    y = add_always_constr(m, label, a, b, rho, K, t)
    add_penalty(m, label, y, obj)
    return y


def build_and_solve(
    spec: stl.STLTerm,
    model_encode_f: Callable[[gModel, int], None],
    spec_obj: float,
    start_robustness_tree: stl.RobustnessTree = None,
    outputflag: int = None,
    numericfocus: int = None,
    threads: int = 4,
    log_files: bool = True,
) -> gModel:
    # print spec
    if spec is not None:
        hd = int(max(0, stl.horizon(spec)) + 1)
    else:
        hd = 0

    m = create_milp("rhc_system")
    logger.debug("Adding system constraints")
    model_encode_f(m, hd)
    # sys_milp.add_sys_constr_x0(m, "d", system, d0, hd, None)
    if spec is not None:
        logger.debug("Adding STL constraints")
        if start_robustness_tree is not None:
            logger.debug("Using starting robustness tree")
        fvar, vbds = add_stl_constr(
            m, "spec", spec, start_robustness_tree=start_robustness_tree
        )
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
            m.numvars, m.numbinvars, m.numconstrs
        )
    )
    m.optimize()
    logger.debug("Finished optimizing")
    if log_files:
        f = open("out_vars.txt", "w")
        for v in m.getVars():
            print(v, f)
        f.close()

    if m.status != GRB.status.OPTIMAL:
        logger.warning("MILP returned status: {}".format(m.status))
    return m
