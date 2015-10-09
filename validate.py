import numpy as np

def cross_validation(data, learn, k=10):
    perm = np.random.permutation(data)
    n = len(data) / k
    folds = [perm[i * n : (i + 1) * n] for i in range(k)]
    folds[-1] = np.append(folds[-1], perm[k * n:])

    missrates = []
    for i in range(k):
        ldata = [x for fold in folds for x in fold]
        classifier = learn(ldata)
        missrates.append(missrate(folds[k], classifier))

    return np.mean(missrates), np.std(missrates)


def missrate(validate, classifier):
    labels = np.array(zip(*validate)[-1])
    test = np.array([classifier.classify(x[:-1]) for x in validate])
    return np.count_nonzero(labels - test) / float(len(labels))
