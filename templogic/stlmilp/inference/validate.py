""" Module with helper functions to run validation tests

Author: Francisco Penedo (franp@bu.edu)

"""
import numpy as np
import pickle


def cross_validation(data, learn, k=10, save=None, disp=False):
    """ Performs a k-fold cross validation test.

    data : a list of labeled traces
           The input data for the cross validation test. It must be a list of
           pairs [trace, label], where the trace is an m by n matrix with the
           last row being the sampling times and the label is 1 or -1.
    learn : a function from data to a classifier
            The learning function. Must accept as a parameter a subset of the
            data and return a classifier. A classifier must be an object with
            a method classify(trace), where trace is defined as in the data
            argument.
    k : integer, optional, defaults to 10
        The number of folds
    save : string, optional
           If specified, the name of a file to save the permutation used to
           split the data.
    disp : boolean, optional, defaults to False
           Toggles the output of debugging information

    """
    p = np.random.permutation(len(data))
    if save is not None:
        with open(save, "wb") as out:
            pickle.dump(p.tolist(), out)

    perm = np.array(data)[p]
    n = len(data) / k
    folds = [perm[i * n : (i + 1) * n] for i in range(k)]
    folds[-1] = np.append(folds[-1], perm[k * n :], axis=0)

    missrates = []
    classifiers = []
    for i in range(k):
        ldata = [x for fold in folds for x in fold]
        classifier = learn(ldata)
        missrates.append(missrate(folds[i], classifier))
        classifiers.append(classifier)
        if disp:
            print(f"Cross validation step {i}")
            print(f"Miss: {missrates[i]}")
            print(classifier.get_formula())

    return np.mean(missrates), np.std(missrates), missrates, classifiers


def missrate(validate, classifier):
    """ Obtains the missrate of a classifier on a given validation set.

    validate : a list of labeled traces
               A validation set. See cross_validation for a description of the
               format
    classifier : an object with a classify method
                 The classifier. See cross_validation for a description

    """
    data, labels = zip(*validate)
    labels = np.array(labels)
    test = np.array([classifier.classify(x) for x in data])
    return np.count_nonzero(labels - test) / float(len(labels))
