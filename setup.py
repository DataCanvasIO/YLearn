import numpy
from Cython.Build import cythonize
from setuptools import setup, Extension

extensions = [
    Extension('ylearn/estimator_model.tree_criterion',
              sources=[f'ylearn/estimator_model/tree_criterion.pyx'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name='ylearn-ext',
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
    zip_safe=False,
)

# setup extensions:
#     python setup.py build_ext --inplace
