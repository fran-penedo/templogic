lltinf - Signal temporal logic data classification
====================

usage: python2 lltinf_cmd.py [-h] [-d D] [-v] [--out-perm f] [-c f] [-o f]
                             [{learn,cv,classify}] file

**Positional arguments:**

*{learn,cv,classify}*

Action to take: 

* 'learn': builds a classifier for the given training set. The resulting stl formula will be printed. 

* 'cv': performs a cross validation test using the given training set. 

* 'classify': classifies a data set using the given classifier (-c must be specified) (default: cv)

*file*

.mat file containing the data

**Optional arguments:**

* -h, --help:
show this help message and exit

*  -d D, --depth D:
maximum depth of the decision tree (default: 3)

*  -v, --verbose:
display more info (default: False)

*  --out-perm f:
if specified, saves the cross validation permutation into f (default: None)

*  -c f, --classifier f:
file containing the classifier (default: None)

*  -o f, --out f:
results from the classification will be stored in MAT format in this file (default: None)

Data set format
--------------------

The data set must be a .mat file with the following format three variables:

* data:
Matrix of real numbers that contains the signals.
Size: Nsignals x Nsamples.
Each row rapresents a signal, each column corresponds to a
sampling time.

* t:
Column vector of real numbers containing the sampling times.
Size: 1 x Nsamples.

* labels:
Column vector of real numbers that contains the labels for the
signals in data.
Size: 1 x Nsignals.
The label is +1 if the corresponging signal belongs to the
positive class C_p.
The label is -1 if the corresponging signal belongs to the
negative class C_N.
Can be omitted when used for classification.

Examples
--------------------

Perform a 10 fold cross-validation test on the Naval data set, limiting the depth of the classiffier to 3 and saving the permutation used:

    $ python2 lltinf_cmd.py -d 3 --out-perm perm.p cv data/Naval/naval_preproc_data_online.mat

Perform a 10 fold cross-validation test on the FuelControl data set, limiting the depth of the classiffier to 3 and saving the permutation used:

    $ python2 lltinf_cmd.py -d 3 --out-perm perm.p cv data/FuelControl/FCdata.mat
