from setuptools import setup, find_packages

config = {
    "description": "Temporal logic tools",
    "url": "",
    "author": "Fran Penedo",
    "author_email": "franp@bu.edu",
    "version": "2.0.1",
    "install_requires": ["numpy>=1.16.4", "pyparsing>=2.1.0"],
    "extras_require": {
        "milp": ["gurobipy>=7.0.1"],
        "inference": ["python-weka-wrapper3>=0.1.7"],
    },
    "packages": find_packages(),
    "scripts": [],
    "name": "templogic",
}

setup(**config)
