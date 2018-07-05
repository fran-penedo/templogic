from lltinf.inference import *
import pprint

def lltinf_simple_test():
    traces = Traces([
        [[1,2,3,4], [1,2,3,4], [1,2,3,4]],
        [[1,2,3,4], [1,2,3,5], [1,2,3,4]]
    ], [1,-1])

    lltinf = LLTInf()
    lltinf.fit(traces)

    print lltinf.tree



