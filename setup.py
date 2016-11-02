try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'STL MILP tools',
    'url': '',
    'author': 'Fran Penedo',
    'author_email': 'franp@bu.edu',
    'version': '0.1.1',
    'install_requires': ['nose'],
    'packages': ['stlmilp'],
    'scripts': [],
    'name': 'stlmilp'
}

setup(**config)
