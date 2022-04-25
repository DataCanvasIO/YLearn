from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
extensions = [
    Extension(
        "tree_criterion",
        ["tree_criterion.pyx"],
        libraries=[],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )]

setup(ext_modules=cythonize(extensions), annotate=True)
