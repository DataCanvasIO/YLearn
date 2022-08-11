# -*- coding:utf-8 -*-
from __future__ import absolute_import

import os
import sys
from glob import glob
from os import path as P

import numpy
from setuptools import setup, Extension, find_packages

my_name = 'ylearn'
excludes_on_windows = ['torch', ]


def read_requirements(file_path='requirements.txt'):
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [x.strip('\n').strip(' ') for x in lines]
    lines = filter(lambda x: len(x) > 0 and not x.startswith('#'), lines)

    is_os_windows = sys.platform.find('win') == 0
    if is_os_windows:
        lines = filter(lambda x: x not in excludes_on_windows, lines)
    lines = list(lines)
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

np_include = numpy.get_include()
pyx_files = glob(f"{my_name}/**/*.pyx", recursive=True)
c_files = glob(f"{my_name}/**/*.cpp", recursive=True)
h_files = glob(f"{my_name}/**/*.h", recursive=True)
my_includes = list(set([P.split(P.abspath(h))[0] for h in h_files]))
build_ext = any(map(lambda s: s == 'build_ext', sys.argv[1:]))

pyx_modules = [P.splitext(f)[0] for f in pyx_files]
c_modules = [P.splitext(f)[0] for f in c_files]

if build_ext:
    c_modules = [f for f in c_modules if f not in pyx_modules]
else:
    pyx_modules = [f for f in pyx_modules if f not in c_modules]
    if pyx_modules:
        msg = f'{len(pyx_modules)} pyx extension(s) without .cpp file found.' \
              f' run "python setup.py build_ext --inplace" to generate .cpp files please.'
        print(msg, file=sys.stderr)

print('cmdline:', ' '.join(sys.argv))
print(f'{my_name}.__version__:' + version)
print('np_version:', numpy.__version__)
print('np_include:', np_include)
print('my_includes:', my_includes)
print('pyx extensions:', pyx_modules)
print('cpp extensions:', c_modules)

c_modules = list(map(lambda f: Extension(f.replace(os.sep, '.'), [f'{f}.cpp'], include_dirs=[np_include]), c_modules))
if pyx_modules:
    from Cython.Build import cythonize

    libraries = []
    if os.name == "posix":
        libraries.append("m")
    pyx_modules = list(map(lambda f: Extension(f.replace(os.sep, '.'), [f'{f}.pyx'],
                                               include_dirs=[np_include] + my_includes,
                                               libraries=libraries,
                                               language="c++",
                                               extra_compile_args=["-O3"],
                                               ),
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
        'ylearn': ['*.txt', '**/**.txt', '**/**.cpp', '**/**.pyx', '**/**.pxd', '**/**.h', ],
    },
    ext_modules=c_modules + pyx_modules,
    include_dirs=[np_include] + my_includes,
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
