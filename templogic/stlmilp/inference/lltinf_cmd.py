"""
Module containing a command line interface to the classification system.

Run with -h option to see the usage help.

Author: Francisco Penedo (franp@bu.edu)

"""
import numpy as np
from scipy.io import loadmat, savemat

# import matplotlib.pyplot as plt
from os import path
import argparse
import os

from .inference import perfect_stop, depth_stop, lltinf, Traces
from ..llt import llt_parser, SimpleModel
from . import validate
from ..stl import satisfies


def load_traces(filename):
    """
    Loads traces' data from a matlab MAT file. The data is stores in 3
    variables called ``data'', ``labels'' and ``t'' which contain the signals'
    values for all dimensions, the class labels (-1 and 1) and the times at
    which the values are sampled.

    filename : string
               The name of a MAT file
    """
    # load data from MAT file
    mat_data = loadmat(filename)
    # add time dimension to signals' data
    dims = list(mat_data["data"].shape)
    dims[1] += 1
    data = np.zeros(dims)
    data[:, : dims[1] - 1, :] = mat_data["data"]
    data[:, dims[1] - 1, :] = mat_data["t"]
    # create list of labels (classes: positive, negative)
    if "labels" in mat_data:
        labels = list(mat_data["labels"][0])
    else:
        labels = None
    # create data structure of traces
    return Traces(signals=data, labels=labels)


def save_traces(traces, filename):
    """
    Saves a Traces object to a MAT file. See load_traces for a description of
    the format.

    traces : a Traces object
    filename : string
               The name of the file to write
    """
    signals, labels = traces.as_list()
    mat_data = {
        "data": signals[:, : (signals.shape[1] - 1), :],
        "t": signals[:, (signals.shape[1] - 1), :],
    }
    if len(labels) > 0:
        mat_data["labels"] = labels

    savemat(filename, mat_data)


# Cross validation test function
def cv_test(matfile, depth=3, out_perm=None, verbose=False):
    traces = load_traces(matfile)
    mean, std, missrates, classifiers = validate.cross_validation(
        zip(*traces.as_list()),
        lltinf_learn(depth, verbose),
        save=out_perm,
        disp=verbose,
    )
    print("Mean: %f" % mean)
    print("Standard Deviation %f" % std)
    for i in range(len(missrates)):
        print("Fold %d - Miss rate: %f" % (i, missrates[i]))
        print(classifiers[i].get_formula())


# Learning wrapper
def lltinf_learn(depth, disp=False):
    return lambda data: lltinf(
        Traces(*zip(*data)),
        depth=depth,
        stop_condition=[perfect_stop, depth_stop],
        disp=disp,
    )


def learn_formula(matfile, depth, verbose=True):
    """
    Obtains and prints a classifier formula from a training set.

    matfile : string
              The name of a MAT file
    depth : integer
            Maximum depth of the formula
    verbose : boolean, optional, defaults to True
              Toggles debugging info
    """
    traces = load_traces(matfile)
    learn = lltinf_learn(depth, verbose)
    classifier = learn(zip(*traces.as_list()))
    print("Classifier:")
    print(classifier.get_formula())


def classify_one(formula, signal):
    return 1 if satisfies(formula, SimpleModel(signal)) else -1


def classify(data, classifier, out):
    """
    Classifies a data set using an STL classifier and writes the classified
    data set to a mat file.

    data : mat file path
           The data set to classify
    classifier : text file path
                 File containing the classifying formula
    out : mat file path
          The output file
    """
    traces = load_traces(data)
    with open(classifier, "r") as c:
        stl = llt_parser().parseString(c.readline())[0]

    labels = [classify_one(stl, signal) for signal in traces.signals]
    save_traces(Traces(traces.signals, labels), out)


def get_argparser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "action",
        choices=["learn", "cv", "classify"],
        nargs="?",
        default="cv",
        help="""
                            action to take:
                            'learn': builds a classifier for the given training
                            set. The resulting stl formula will be printed.
                            'cv': performs a cross validation test using the
                            given training set.
                            'classify': classifies a data set using the given
                            classifier (-c must be specified)
                            """,
    )
    parser.add_argument(
        "-d",
        "--depth",
        metavar="D",
        type=int,
        default=3,
        help="maximum depth of the decision tree",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="display more info"
    )
    parser.add_argument("file", help=".mat file containing the data")
    parser.add_argument(
        "--out-perm",
        metavar="f",
        default=None,
        help="if specified, saves the cross validation permutation into f",
    )
    parser.add_argument(
        "-c",
        "--classifier",
        metavar="f",
        default=None,
        help="file containing the classifier",
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="f",
        default=None,
        help="results from the classification will be stored in MAT format in this file",
    )
    return parser


def get_path(f):
    return path.join(os.getcwd(), f)


if __name__ == "__main__":
    args = get_argparser().parse_args()
    if args.action == "learn":
        learn_formula(get_path(args.file), args.depth)
    elif args.action == "cv":
        cv_test(get_path(args.file), args.depth, get_path(args.out_perm))
    elif args.action == "classify":
        classify(get_path(args.file), get_path(args.classifier), get_path(args.out))
    else:
        raise Exception("Action not implemented")
