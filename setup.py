from __future__ import print_function
import sys
from setuptools import setup, Extension
from Cython.Distutils import build_ext

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)


"""
_tree = Extension('id3._tree',
                  sources=['id3/_tree.pyx'],
                  include_dirs=[numpy.get_include()])
"""
_splitter = Extension('id3._splitter',
                      sources=['id3/_splitter.pyx'],
                      include_dirs=[numpy.get_include()])


config = {
        'name': 'decision-tree-id3',
        'version': '0.0.1',
        'description': 'A scikit-learn compatible package for ID3.',
        'author': 'Daniel Pettersson, Otto Nordander',
        'packages': ['id3'],
        'install_requires': INSTALL_REQUIRES,
        'author_email': 'svaante@gmail.com, otto.nordander@gmail.com',
        'cmdclass': {'build_ext': build_ext},
        'ext_modules': [_splitter]
        }

if __name__ == '__main__':
    setup(**config)
