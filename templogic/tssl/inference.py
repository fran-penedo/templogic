import logging
from typing import Iterable, Callable, Tuple, Sequence, Optional
from functools import partial

import numpy as np
import attr

_HANDLERS = list(logging.getLogger().handlers)
_LEVEL = logging.getLogger().getEffectiveLevel()

import weka  # type: ignore
from weka.classifiers import Classifier  # type: ignore
import javabridge  # type: ignore

# weka executes logging.basicConfig, which we revert here
# Possibly overkill if basicConfig does nothing when there is a handler already
logging.getLogger().setLevel(_LEVEL)
logging.getLogger().handlers = _HANDLERS

from . import tssl, quadtree

logger = logging.getLogger(__name__)

IRIS = "/usr/share/data/weka/iris.arff"


def start_jvm() -> None:
    weka.core.jvm.start(max_heap_size="32m")


def stop_jvm() -> None:
    weka.core.jvm.stop()


@attr.s(auto_attribs=True)
class Instances(object):
    instances: weka.core.dataset.Instances
    depth: int

    def __getattr__(self, attr):
        return getattr(self.instances, attr)

    def __iter__(self):
        return iter(self.instances)


class TSSLInference(Classifier):
    """Builds a TSSL formula that classifies the data

    Uses Ripper algorithm (weka's JRip) to build a rule classifier, then translates
    rules into a TSSL formula
    """

    _form: tssl.TSSLTerm

    def __init__(self) -> None:
        weka.core.jvm.start()
        super().__init__(classname="weka.classifiers.rules.JRip")
        fields = javabridge.get_env().get_object_array_elements(
            javabridge.call(
                self.jwrapper.class_wrapper.o,
                "getDeclaredFields",
                "()[Ljava/lang/reflect/Field;",
            )
        )
        for field in fields:
            name = javabridge.call(field, "getName", "()Ljava/lang/String;")
            if name == "m_Class":
                self._class_attribute_field = field
                break

        if not hasattr(self, "_class_attribute_field"):
            raise Exception("JRip java class does not have m_Class field")

    def get_tssl_formula(self) -> tssl.TSSLTerm:
        if self._form is None:
            raise Exception("Classifier has not been built")
        else:
            return self._form

    def build_classifier(self, data: Instances, valid_class: str = "0") -> None:
        super().build_classifier(data)
        logger.debug("Built JRip classifier")
        logger.debug(self)
        self._form = parse_rules(
            self.jwrapper.getRuleset(), data.depth, self.class_attribute, valid_class
        )

    @property
    def class_attribute(self):
        return javabridge.JWrapper(
            javabridge.call(
                self._class_attribute_field,
                "get",
                "(Ljava/lang/Object;)Ljava/lang/Object;",
                self.jwrapper,
            )
        )


def parse_antd(antd, depth: int) -> tssl.TSSLTerm:
    # Attribute names are x{index}
    index = int(antd.getAttr().name()[1:])
    label = quadtree._index_to_label(index, depth)
    value = antd.getSplitPoint()

    rel = tssl.Relation.LE if antd.getAttrValue() == 0 else tssl.Relation.GE
    a = [0 for i in range(label[0] + 1)]
    a[label[0]] = 1
    cur: tssl.TSSLTerm = tssl.TSSLPred(a, value, rel)

    label.reverse()
    for l in label[:-1]:
        ds = [tssl.Direction(l)]
        cur = tssl.TSSLExistsNext(ds, cur)

    return cur


def parse_rule(rule, depth: int) -> tssl.TSSLTerm:
    and_args = [parse_antd(antd, depth) for antd in rule.getAntds()]
    if len(and_args) == 0:
        # FIXME might need this one
        raise Exception("Empty antecedents")
    elif len(and_args) == 1:
        return and_args[0]
    else:
        return tssl.TSSLAnd(and_args)


def parse_rules(rules, depth: int, class_attr, valid_class: str) -> tssl.TSSLTerm:
    rules = list(rules)
    assert len(rules[-1].getAntds()) == 0
    rule_forms = [parse_rule(r, depth) for r in rules[:-1]]
    negs = [tssl.TSSLNot(form.copy()) for form in rule_forms]
    or_args = [
        tssl.TSSLAnd([rule_forms[i]] + [n.copy() for n in negs[:i]])
        for i in range(len(rule_forms))
        if class_attr.value(rules[i].getConsequent()) == valid_class
    ]
    if len(or_args) == 0:
        if len(negs) > 0:
            return tssl.TSSLAnd(negs)
        elif class_attr.value(rules[-1].getConsequent()) == valid_class:
            return tssl.TSSLTop()
        else:
            return tssl.TSSLBottom()
    elif len(or_args) == 1:
        return or_args[0]
    else:
        return tssl.TSSLOr(or_args)


def build_dataset(
    imgs: np.ndarray,
    labels: Iterable,
    fun: Callable[[Sequence[float]], float] = partial(np.mean, axis=0),
) -> Instances:
    qts = [quadtree.QuadTree.from_matrix(img, fun) for img in imgs]
    data = [qt.flatten() + [label] for qt, label in zip(qts, labels)]
    dataset = _create_dataset(data, qts[0].depth())
    for i in range(dataset.num_attributes - 1):
        dataset.jwrapper.renameAttribute(i, _att_label(dataset.attribute(i).name))

    return dataset


def _att_label(label: str) -> str:
    index = int(label[1:]) - 1
    return "x{}".format(str(index))


def _create_dataset(data: np.ndarray, depth: int) -> Instances:
    weka.core.jvm.start()
    dataset = weka.core.dataset.create_instances_from_lists(data)
    dataset.class_is_last()
    filt = weka.filters.Filter(
        "weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"]
    )
    filt.inputformat(dataset)
    return Instances(filt.filter(dataset), depth)


# Test images
def build_spirals(
    shape: Tuple[int, int] = (16, 16), maxspirals: int = 5, spiralsize: int = 2
) -> np.ndarray:
    """ Builds an image with `shape` and a random amount of spirals

    The image will contain at least one spiral.
    Spiral diameter is controlled by `spiralsize`
    """
    size = spiralsize + 2
    img = np.random.binomial(1, 0.5, shape)
    nspirals = 1 + np.random.choice(maxspirals)
    compatible = False
    while not compatible:
        compatible = True
        xs = np.random.choice(shape[0] - size, size=nspirals)
        ys = np.random.choice(shape[1] - size, size=nspirals)
        vs = list(zip(xs, ys))
        try:
            add_spirals(img, vs, spiralsize)
        except IncompatibleSpiralsException:
            compatible = False

    return img.reshape(shape + (1,))


def build_rotating_spirals(
    shape: Tuple[int, int, int] = (4, 16, 16), maxspirals: int = 5, spiralsize: int = 2
) -> np.ndarray:
    size = spiralsize + 2
    img = np.random.binomial(1, 0.5, shape)
    nspirals = 1 + np.random.choice(maxspirals)
    compatible = False
    npictures, xshape, yshape = shape
    while not compatible:
        compatible = True
        xs = np.random.choice(xshape - size, size=nspirals)
        ys = np.random.choice(yshape - size, size=nspirals)
        vs = list(zip(xs, ys))
        rots = np.random.choice(4, nspirals)
        try:
            for i in range(npictures):
                add_spirals(img[i], vs, spiralsize, rots)
                # Rotation magic, just for fun
                rots = (1 + (rots % 2) * 2 - rots // 2) % 4
        except IncompatibleSpiralsException:
            compatible = False

    return img.reshape(shape + (1,))


class IncompatibleSpiralsException(Exception):
    pass


def add_spirals(
    img: np.ndarray,
    vs: Sequence[Tuple[int, int]],
    spiralsize: int,
    rots: Optional[Sequence[int]] = None,
) -> np.ndarray:
    if spiralsize == 2:
        _add_spiral = add_spiral
    elif spiralsize == 3:
        _add_spiral = add_spiral_3
    size = spiralsize + 2
    rots = rots if rots is not None else np.random.choice(4, len(vs))
    for i, v in enumerate(vs):
        for v2 in vs[i + 1 :]:
            if collision(v, v2, size):
                raise IncompatibleSpiralsException()
    for v in vs:
        _add_spiral(img, v)


def add_spiral(img, v, rot=None):
    """ Adds a spiral to `img` at point `v`

    A spiral any rotation of

    0 0 0 0
    0 1 1 0
    0 1 0 0
    0 0 0 0

    """
    spiral = np.zeros((4, 4))
    spiral[1:-1, 1:-1] = 1
    i = rot if rot is not None else np.random.choice(4)
    spiral[1 + (i // 2), 1 + (i % 2)] = 0
    img[v[0] : v[0] + 4, v[1] : v[1] + 4] = spiral


def add_spiral_3(img, v, rot=None):
    """ Adds a 3x3 spiral to `img` at point `v`

    A 3x3 spiral is any rotation of

    0 0 0 0 0
    0 1 1 1 0
    0 1 1 1 0
    0 0 1 1 0
    0 0 0 0 0

    """
    spiral = np.zeros((5, 5))
    spiral[1:-1, 1:-1] = 1
    i = rot if rot is not None else np.random.choice(4)
    spiral[1 + 2 * (i // 2), 1 + 2 * (i % 2)] = 0
    img[v[0] : v[0] + 5, v[1] : v[1] + 5] = spiral


def collision(v, v2, size=4):
    """ Detects a collision between two squares of the same size

    """

    def _collision_oneside(a, b):
        def _in(i):
            return a[i] >= b[i] and a[i] <= b[i] + size - 1

        return _in(0) and _in(1)

    return _collision_oneside(v, v2) or _collision_oneside(v, v2)
