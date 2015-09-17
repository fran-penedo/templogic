from llt import *
import pprint

def primitives_test():
    signals = [
        [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
        [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
    ]
    prims = make_llt_primitives(signals)
    set_llt_pars(prims[0], 1, 2, 3, 4)
    pprint.pprint(prims)

    pprint.pprint(prims[0].copy())

