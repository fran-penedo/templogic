Building lltinf
====================

To build lltinf, you will need:

* Python 2.7
* pip 7.0.1 or newer
* optionally, virtualenv 13.1.2 or newer

The package should work in any of the major platforms, although only Linux is supported.

Installing dependencies
--------------------

If you want to install the dependencies on a virtual environment, from the top level of the package run:

    $ virtualenv -p python2 venv
    $ source venv/bin/activate

To install the dependencies, run:

    $ pip install -r requirements.txt

Testing the installation
--------------------

Check if everything is correctly installed by running:

    $ nosetests
