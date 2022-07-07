# -*- coding:utf-8 -*-
from __future__ import absolute_import

import os
import sys
from glob import glob
from os import path as P

import numpy
from setuptools import setup, Extension, find_packages

my_name = 'ylearn'


def read_requirements(file_path='requirements.txt'):
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [x.strip('\n').strip(' ') for x in lines]
    lines = list(filter(lambda x: len(x) > 0 and not x.startswith('#'), lines))

    return lines


def read_extra_requirements():
    import glob
    import re

    extra = {}

    for file_name in glob.glob('requirements-*.txt'):
        key = re.search('requirements-(.+).txt', file_name).group(1)
        req = read_requirements(file_name)
        if req:
            extra[key] = req

    if extra and 'all' not in extra.keys():
        extra['all'] = sorted({v for req in extra.values() for v in req})

    return extra


try:
    execfile
except NameError:
    def execfile(fname, globs, locs=None):
        locs = locs or globs
        exec(compile(open(fname).read(), fname, "exec"), globs, locs)

HERE = P.dirname((P.abspath(__file__)))
version_ns = {}
execfile(P.join(HERE, my_name, '_version.py'), version_ns)
version = version_ns['__version__']
print("__version__=" + version)

np_include = numpy.get_include()
pyx_files = glob(f"{my_name}/**/*.pyx", recursive=True)
c_files = glob(f"{my_name}/**/*.cpp", recursive=True)
build_ext = any(map(lambda s: s == 'build_ext', sys.argv[1:]))

if build_ext:
    pyx_modules = [P.splitext(f)[0] for f in pyx_files]
    c_modules = [P.splitext(f)[0] for f in c_files]
    c_modules = [f for f in c_modules if f not in pyx_modules]
else:
    c_modules = [P.splitext(f)[0] for f in c_files]
    for pf in pyx_files:
        if P.splitext(pf)[0] not in c_modules:
            raise FileNotFoundError(f'Not found c file for {pf}, '
                                    f'run "python setup.py build_ext --inplace" to generate c files.')
    pyx_modules = []
print('pyx extensions:', pyx_modules)
print('cpp extensions:', c_modules)
print('np_include', np_include)

c_modules = list(map(lambda f: Extension(f.replace(os.sep, '.'), [f'{f}.cpp'], include_dirs=[np_include]), c_modules))
if pyx_modules:
    from Cython.Build import cythonize

    pyx_modules = list(map(lambda f: Extension(f.replace(os.sep, '.'), [f'{f}.pyx'],
                                               include_dirs=[np_include],
                                               language="c++"),
                           pyx_modules))
    pyx_modules = cythonize(pyx_modules, compiler_directives={'language_level': "3"})

MIN_PYTHON_VERSION = '>=3.6.*'

long_description = open('README.md', encoding='utf-8').read()

requires = read_requirements()
extras_require = read_extra_requirements()
print('requirements:', requires)

setup(
    name=my_name,
    version=version,
    description='A python package for causal inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/DataCanvasIO/YLearn',
    author='DataCanvas Community',
    author_email='yangjian@zetyun.com',
    license='Apache License 2.0',
    install_requires=requires,
    python_requires=MIN_PYTHON_VERSION,
    extras_require=extras_require,
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    packages=find_packages(exclude=('docs', 'tests', 'example_usages')),
    package_data={
        'ylearn': ['*.txt', '**/**.txt', ],
    },
    ext_modules=c_modules + pyx_modules,
    include_dirs=[np_include],
    zip_safe=False,
    # entry_points={
    #     # 'console_scripts': [
    #     #     'why = ylearn.foo:main',
    #     # ],
    # },
    include_package_data=True,
)

# setup extensions:
#     python setup.py build_ext --inplace
