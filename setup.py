try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    "description": "STL MILP tools",
    "url": "",
    "author": "Fran Penedo",
    "author_email": "franp@bu.edu",
    "version": "1.0.1",
    "install_requires": ["numpy>=1.16.4", "pyparsing>=2.1.0", "gurobipy>=7.0.2"],
    "packages": ["stlmilp"],
    "scripts": [],
    "name": "stlmilp",
}

setup(**config)
