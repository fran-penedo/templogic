try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'STL inference',
    'url': '',
    'author': 'Fran Penedo',
    'author_email': 'franp@bu.edu',
    'version': '0.1.1',
    'install_requires': [],
    'packages': ['lltinf'],
    'scripts': [],
    'name': 'lltinf'
}

setup(**config)
