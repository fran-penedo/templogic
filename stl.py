import operator

# Operator constants
EXPR = 0
NOT = 1
AND = 2
OR = 3
NEXT = 4
ALWAYS = 5
EVENTUALLY = 6

LE = operator.le
GT = operator.gt


class Signal(object):

    def __init__(self, labels, f):
        self._labels = labels
        self._f = f

    def signal(self, model, t):
        vs = [model.getVarByName(l(t)) for l in self._labels]
        if any(var is None for var in vs):
            return None
        else:
            return self._f(vs)


class Formula(object):

    def __init__(self, operator, args, bounds=[0, 0]):
        self._op = operator
        self._args = args
        self._bounds = bounds

    def _hexpr(self):
        return 0

    def _hnot(self):
        return 0

    def _hand(self):
        return max(map(lambda f: f.horizon(), self.args))

    def _hor(self):
        return self._hand()

    def _halways(self):
        return self.bounds[1] + self.args[0].horizon()

    def _hnext(self):
        return 1 + self.args[0].horizon()

    def _heventually(self):
        return self._halways()

    def horizon(self):
        return {
            EXPR: self._hexpr,
            NOT: self._hnot,
            AND: self._hand,
            OR: self._hor,
            NEXT: self._hnext,
            ALWAYS: self._halways,
            EVENTUALLY: self._heventually
        }[self.op]()

    @property
    def op(self):
        return self._op

    @op.setter
    def op(self, value):
        self._op = value

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = value

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = value

def robustness(formula, model, t=0):
    return {
        EXPR: lambda: formula.args[0].signal(model, t),
        NOT: lambda: -robustness(formula.args[0], model, t),
        AND: lambda: min(map(lambda f: robustness(f, model, t), formula.args)),
        OR: lambda: max(map(lambda f: robustness(f, model, t), formula.args)),
        NEXT: lambda: robustness(formula.args[0], model, t + 1),
        ALWAYS: lambda: min(map(
            lambda j: robustness(formula.args[0], model, t + j),
            range(formula.bounds[0], formula.bounds[1] + 1))),
        EVENTUALLY: lambda: max(map(
            lambda j: robustness(formula.args[0], model, t + j),
            range(formula.bounds[0], formula.bounds[1] + 1)))
    }[formula.op]()

def satisfies(formula, model, t=0):
    return robustness(formula, model, t) >= 0
