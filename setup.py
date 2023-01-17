# -*- coding:utf-8 -*-
from __future__ import absolute_import

import os
import sys
from glob import glob
from os import path as P

import numpy
from setuptools import setup, Extension, find_packages

my_name = 'ylearn'
home_url = 'https://github.com/DataCanvasIO/YLearn'

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


def read_description(file_path='README.md', image_root=f'{home_url}/raw/main', ):
    import os
    import re

    def _encode_image(m):
        assert len(m.groups()) == 3

        pre, src, post = m.groups()
        src = src.rstrip().lstrip()

        remote_src = os.path.join(image_root, os.path.relpath(src))
        return f'{pre}{remote_src}{post}'

    desc = open(file_path, encoding='utf-8').read()

    # remove QRCode
    desc = '\n'.join([line for line in desc.splitlines() if line.find('QRcode') < 0])

    # substitute html image
    desc = re.sub(r'(<img\s+src\s*=\s*\")(./fig/[^"]+)(\")', _encode_image, desc)

    # substitute markdown image
    desc = re.sub(r'(\!\[.*\]\()(./fig/.+)(\))', _encode_image, desc)

    return desc


def find_extensions(*base_dirs):
    pyx = []
    cpp = []
    h = []
    for d in base_dirs:
        if d.endswith('.pyx'):
            pyx.append(d)
        elif d.endswith('.cpp'):
            cpp.append(d)
        elif d.endswith('.h'):
            h.append(d)
        else:
            pyx.extend(glob(f"{d}/**/*.pyx", recursive=True))
            cpp.extend(glob(f"{d}/**/*.cpp", recursive=True))
            h.extend(glob(f"{d}/**/*.h", recursive=True))
    return pyx, cpp, h


def download(url, file_path):
    import shutil
    import urllib3

    http = urllib3.PoolManager()
    with http.request('GET', url, preload_content=False) as r, open(file_path, 'wb') as out_file:
        if r.status != 200:
            raise ConnectionError(f'[{r.status}] Failed to request {url}')
        shutil.copyfileobj(r, out_file)


def extract(tar_path, extract_path='.'):
    """
    see: https://www.codegrepper.com/code-examples/python/extract+tgz+files+in+python
    """
    import tarfile

    tar = tarfile.open(tar_path, 'r')
    for item in tar:
        tar.extract(item, extract_path)
        if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
            extract(item.name, "./" + item.name[:item.name.rfind('/')])


def prepare_eigen3(ver='3.4.0', target_path='./downloads'):
    eigen3_path = os.environ.get('EIGEN3_PATH', None)

    if eigen3_path is None:
        tarball = f'eigen-{ver}'
        eigen3_path = f'{target_path}/{tarball}'
        if (not os.path.exists(eigen3_path)) or len(dir(f'{eigen3_path}/*')) == 0:
            os.makedirs(target_path, exist_ok=True)

            tarball_file = f'{target_path}/{tarball}.tar.gz'
            if (not os.path.exists(tarball_file)) or os.path.getsize(tarball_file) < 1:
                print(f'download {tarball} ...')
                url = f'https://gitlab.com/libeigen/eigen/-/archive/{ver}/{tarball}.tar.gz'
                download(url, tarball_file)

            print(f'extract {tarball} ...')
            extract(tarball_file, target_path)

    assert os.path.exists(eigen3_path), f'Not found path EIGEN3_PATH: {eigen3_path}'

    print(f'using EIGEN3_PATH: {eigen3_path}')
    return eigen3_path


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
# pyx_files = glob(f"{my_name}/**/*.pyx", recursive=True)
# c_files = glob(f"{my_name}/**/*.cpp", recursive=True)
# h_files = glob(f"{my_name}/**/*.h", recursive=True)
pyx_files, c_files, h_files = find_extensions(*os.environ.get('EXT_PATH', my_name).split(','))
my_includes = list(set([P.split(P.abspath(h))[0] for h in h_files]))
my_includes = my_includes + [prepare_eigen3()]
build_ext = any(map(lambda s: s == 'build_ext', sys.argv[1:]))

pyx_modules = [P.splitext(f)[0] for f in pyx_files]
c_modules = [P.splitext(f)[0] for f in c_files]
c_modules = [f for f in c_modules if not f.endswith('_lib')]  # no-extension

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

# long_description = open('README.md', encoding='utf-8').read()
long_description = read_description('README.md')

requires = read_requirements()
extras_require = read_extra_requirements()
print('requirements:', requires)

setup(
    name=my_name,
    version=version,
    description='A python package for causal inference',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=home_url,
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
        'Programming Language :: Python :: 3.10',
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

# build all extensions:
#     python setup.py build_ext --inplace
#
# build extension 'sklearn_ex' and 'grf._criterion':
#     EXTPATH=ylearn/sklearn_ex,ylearn/estimator_model/_generalized_forest/tree/_criterion.pyx  python setup.py build_ext --inplace
