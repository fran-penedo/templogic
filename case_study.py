import numpy as np
from scipy.io import loadmat
#import matplotlib.pyplot as plt
from os import path

from lltinf import perfect_stop, depth_stop, lltinf, Traces

SIMPLEDS=path.join('data', 'SimpleDS', 'SimpleDS.mat')
SIMPLEDS2=path.join('data', 'SimpleDS2', 'simpleDS2.mat')

def load_traces(filename):
    ''' Loads traces' data from a matlab MAT file. The data is stores in 3
    variables called ``data'', ``labels'' and ``t'' which contain the signals'
    values for all dimensions, the class labels (-1 and 1) and the times at
    which the values are sampled.
    '''
    # load data from MAT file
    mat_data =  loadmat(filename)
    # add time dimension to signals' data
    dims = list(mat_data['data'].shape)
    dims[1] += 1
    data = np.zeros(dims)
    data[:, :dims[1]-1, :] = mat_data['data']
    data[:, dims[1]-1, :] = mat_data['t']
    # create list of labels (classes: positive, negative)
    labels = list(mat_data['labels'][0])
    # create data structure of traces
    return Traces(signals=data, labels=labels)

def lltinf_simple_test():
    '''Simple case study.'''
    traces = load_traces(SIMPLEDS)
    print traces.signals.shape
    print len(traces.labels)
    # run classification algorithm
    lltinf(traces, depth=1, stop_condition=[perfect_stop, depth_stop])

def lltinf_simple2_test():
    '''Simple case study.'''
    traces = load_traces(SIMPLEDS2)
    print traces.signals.shape
    print len(traces.labels)
    # run classification algorithm
    tree = lltinf(traces, depth=2, stop_condition=[perfect_stop, depth_stop])
    print tree.get_formula()

def lltinf_naval_test():
    '''Naval vessel case study.'''
    traces = load_traces(r'..\Test Cases\Naval\naval_preproc_data_online.mat')
    print traces.signals.shape
    print len(traces.labels)
    # run classification algorithm
    lltinf(traces, depth=1, stop_condition=[perfect_stop, depth_stop])


if __name__ == '__main__':
    #lltinf_simple_test()
    lltinf_simple2_test()
    #lltinf_naval_test()
