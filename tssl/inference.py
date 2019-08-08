import logging

import numpy as np  # type: ignore

ROOT_LOGGER_LEVEL = logging.getLogger().getEffectiveLevel()
import weka  # type: ignore
from weka.classifiers import Classifier  # type: ignore
import javabridge  # type: ignore

logging.getLogger().setLevel(ROOT_LOGGER_LEVEL)

from tssl import tssl, quadtree

LOGGER = logging.getLogger(__name__)

IRIS = "/usr/share/data/weka/iris.arff"


def start_jvm():
    weka.core.jvm.start()


def stop_jvm():
    weka.core.jvm.stop()


class TSSLInference(Classifier):
    """Builds a TSSL formula that classifies the data

    Uses Ripper algorithm (weka's JRip) to build a rule classifier, then translates
    rules into a TSSL formula
    """

    def __init__(self):
        """TODO: to be defined1.

        """
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

    def get_tssl_formula(self):
        """TODO: Docstring for get_tssl.

        Returns
        -------
        TODO

        """
        if self._form is None:
            raise Exception("Classifier has not been built")
        else:
            return self._form

    def build_classifier(self, data, depth, valid_class="0"):
        """TODO: Docstring for build_classifier.

        Parameters
        ----------
        data : TODO
        depth : TODO
        valid_class

        Returns
        -------
        TODO

        """
        super().build_classifier(data)
        LOGGER.debug("Built JRip classifier")
        LOGGER.debug(self)
        self._form = parse_rules(
            self.jwrapper.getRuleset(), depth, self.class_attribute, valid_class
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


def parse_antd(antd, depth):
    # Attribute names are x{index}
    index = int(antd.getAttr().name()[1:])
    label = quadtree._index_to_label(index, depth)
    value = antd.getSplitPoint()

    rel = tssl.Relation.LE if antd.getAttrValue() == 0 else tssl.Relation.GE
    a = [0 for i in range(label[0] + 1)]
    a[label[0]] = 1
    cur = tssl.TSSLPred(a, value, rel)

    label.reverse()
    for l in label[:-1]:
        ds = [tssl.Direction(l)]
        cur = tssl.TSSLExistsNext(ds, cur)

    return cur


def parse_rule(rule, depth):
    and_args = [parse_antd(antd, depth) for antd in rule.getAntds()]
    if len(and_args) == 0:
        # FIXME might need this one
        raise Exception("Empty antecedents")
    elif len(and_args) == 1:
        return and_args[0]
    else:
        return tssl.TSSLAnd(and_args)


def parse_rules(rules, depth, class_attr, valid_class):
    """TODO: Docstring for parse_rules.

    Parameters
    ----------
    rules : TODO
    depth : TODO

    Returns
    -------
    TODO

    """
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


def build_dataset(imgs, labels, fun=np.mean):
    """TODO: Docstring for build_dataset.

    Parameters
    ----------
    imgs : TODO
    labels : TODO
    fun : TODO

    Returns
    -------
    TODO

    """
    qts = [quadtree.QuadTree.from_matrix(img, fun) for img in imgs]
    data = [qt.flatten() + [label] for qt, label in zip(qts, labels)]
    dataset = _create_dataset(data)
    for i in range(dataset.num_attributes - 1):
        dataset.jwrapper.renameAttribute(i, _att_label(dataset.attribute(i).name))

    return dataset


def _att_label(label):
    index = int(label[1:]) - 1
    return "x{}".format(str(index))


def _create_dataset(data):
    dataset = weka.core.dataset.create_instances_from_lists(data)
    dataset.class_is_last()
    filt = weka.filters.Filter(
        "weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"]
    )
    filt.inputformat(dataset)
    return filt.filter(dataset)


# Test images
def build_spirals(shape=(16, 16), maxspirals=5):
    img = np.random.binomial(1, 0.5, shape)
    nspirals = np.random.choice(maxspirals + 1)
    compatible = False
    while not compatible:
        compatible = True
        xs = np.random.choice(shape[0] - 4, size=nspirals)
        ys = np.random.choice(shape[1] - 4, size=nspirals)
        vs = list(zip(xs, ys))
        for i, v in enumerate(vs):
            for v2 in vs[i + 1 :]:
                if collision(v, v2):
                    compatible = False
                    break
            add_spiral(img, v)

    return img.reshape(shape + (1,))


def add_spiral(img, v):
    spiral = np.zeros((4, 4))
    spiral[1:-1, 1:-1] = 1
    i = np.random.choice(4)
    spiral[1 + (i // 2), 1 + (i % 2)] = 0
    img[v[0] : v[0] + 4, v[1] : v[1] + 4] = spiral


def collision(v, v2):
    def _collision_oneside(a, b):
        def _in(i):
            return a[i] >= b[i] and a[i] <= b[i] + 3

        return _in(0) and _in(1)

    return _collision_oneside(v, v2) or _collision_oneside(v, v2)
