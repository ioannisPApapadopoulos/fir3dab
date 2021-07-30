# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import sys
import os
import re
import glob


version = re.findall('__version__ = "(.*)"',
                     open('fir3dab/__init__.py', 'r').read())[0]

packages = [
    "fir3dab",
    ]

CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

# TODO: This is cumbersome and prone to omit something
demofiles = glob.glob(os.path.join("examples", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.xml*"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.geo"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.xml*"))

# Don't bother user with test files
[demofiles.remove(f) for f in demofiles if "test_" in f]

setup(name="fir3dab",
      version=version,
      author="Ioannis Papadopoulos",
      author_email="papadopoulos@maths.ox.ac.uk",
      url="https://github.com/ioannisPApapadopoulos/fir3dab",
      description="Deflated Barrier Method",
      long_description="--",
      classifiers=classifiers,
      license="GNU LGPL v3 or later",
      packages=packages,
      package_dir={"fir3dab": "fir3dab"},
      data_files=[(os.path.join("share", "fir3dab", os.path.dirname(f)), [f])
                  for f in demofiles],
    )
