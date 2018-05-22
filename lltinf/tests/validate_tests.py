from lltinf.validate import *
import numpy as np
import numpy.testing as test

def missrate_test():
    val = zip(range(5), [1,1,1, -1, -1])
    class classifier:
        def classify(self, x):
            return 1 if x < 2 else -1

    test.assert_almost_equal(missrate(val, classifier()), 1.0/5)
