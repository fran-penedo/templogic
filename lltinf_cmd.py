import numpy as np
from scipy.io import loadmat, savemat
#import matplotlib.pyplot as plt
from os import path
import validate
import argparse
import os

from lltinf import perfect_stop, depth_stop, lltinf, Traces

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
    if 'labels' in mat_data:
        labels = list(mat_data['labels'][0])
    else:
        labels = None
    # create data structure of traces
    return Traces(signals=data, labels=labels)


def save_traces(traces, filename):
    signals, labels = traces.as_list()
    mat_data = {
        'data': signals[:, :(signals.shape[1] - 1), :],
        't': signals[:, (signals.shape[1] - 1), :],
    }
    if len(labels) > 0:
        mat_data['labels'] = labels

    savemat(filename, mat_data)


def cv_test(matfile, depth=3, out_perm=None):
    traces = load_traces(matfile)
    mean, std, missrates, classifiers = \
        validate.cross_validation(zip(*traces.as_list()), lltinf_learn(depth),
                                  save=out_perm)
    print "Mean: %f" % mean
    print "Standard Deviation %f" % std
    for i in range(len(missrates)):
        print "Fold %d - Miss rate: %f" % (i, missrates[i])
        print classifiers[i].get_formula()


def lltinf_learn(depth):
    return lambda data: lltinf(Traces(*zip(*data)), depth=depth,
                               stop_condition=[perfect_stop, depth_stop])

def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-d', '--depth', metavar='D', type=int, nargs=1,
                        default=3, help='maximum depth of the decision tree')
    parser.add_argument('file', nargs=1, help='.mat file containing the data')
    parser.add_argument('--out-perm', metavar='f', nargs=1, default=None,
                        help='if specified, saves the cross validation permutation into f')
    return parser

if __name__ == '__main__':
    args = get_argparser().parse_args()
    cv_test(path.join(os.getcwd(), args.file[0]), args.depth[0],
            path.join(os.getcwd(), args.out_perm[0]))
