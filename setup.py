from setuptools import setup, find_packages

config = {
    "description": "Temporal logic tools",
    "url": "",
    "author": "Fran Penedo",
    "author_email": "franp@bu.edu",
    "version": "2.0.2",
    "install_requires": ["numpy>=1.16.4", "pyparsing>=2.1.0", "attrs>=19.1.0"],
    "extras_require": {
        "milp": ["gurobipy>=7.0.1"],
        "inference": ["python-weka-wrapper3>=0.1.7", "scipy>=1.3.1"],
    },
    "packages": find_packages(),
    "scripts": [],
    "entry_points": {
        "console_scripts": ["lltinf=templogic.stlmilp.inference.lltinf_cmd:main"]
    },
    "name": "templogic",
}

setup(**config)
