import numpy as np

def cross_validation(data, learn, k=10):
    perm = np.random.permutation(data)
    n = len(data) / k
    folds = [perm[i * n : (i + 1) * n] for i in range(k)]
    folds[-1] = np.append(folds[-1], perm[k * n:], axis=0)

    missrates = []
    classifiers = []
    for i in range(k):
        print "Cross validation step %d" % i
        ldata = [x for fold in folds for x in fold]
        classifier = learn(ldata)
        missrates.append(missrate(folds[i], classifier))
        classifiers.append(classifier)

    return np.mean(missrates), np.std(missrates), missrates, classifiers

def missrate(validate, classifier):
    data, labels = zip(*validate)
    labels = np.array(labels)
    test = np.array([classifier.classify(x) for x in data])
    return np.count_nonzero(labels - test) / float(len(labels))
