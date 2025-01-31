#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup
from sphinx.setup_command import BuildDoc

def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()

cmdclass = {'build_sphinx': BuildDoc}
name = 'mdfeature'
release = '0.1.0'

#    cmdclass=cmdclass,
#    command_options={
#        'build_doc': {
#            'project': ('setup.py', name),
#            'release': ('setup.py', release),
#            'source_dir': ('setup.py', 'doc')}},

setup(
    name=name,
    version=release,
    license='GPL license',
    description='Library for automatics selection of collective variables.',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Zofia Trstanova',
    author_email='zofia.trstanova [at symbol] gmail.com',
    url='https://github.com/ZofiaTr/mdfeature',
    packages=find_packages('src'),
    package_dir={'':'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering'
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    install_requires=[
        'numpy', 'scipy', 'scikit-learn', 'matplotlib',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
)
#    entry_points={
#        'console_scripts': [
#            'pyDiffMap = pyDiffMap.cli:main',
#        ]
#    },


#TODO:
# Add notes on installation - conda environment and pip install pyDiffMap