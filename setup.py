#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Setup file for PyScaffold.
    Important note: Since PyScaffold is self-using and depends on
    setuptools-scm, it is important to run `python setup.py egg_info` after
    a fresh checkout. This will generate some critically needed data.
"""
import sys

from setuptools import setup

__author__ = "Florian Wilhelm"
__copyright__ = "Blue Yonder"
__license__ = "new BSD"


def setup_package():
    needs_sphinx = {'build_sphinx', 'upload_docs'}.intersection(sys.argv)
    sphinx = ['sphinx'] if needs_sphinx else []
    setup(setup_requires=['six', 'pyscaffold>=2.5a0,<2.6a0'] + sphinx,
          use_pyscaffold=True)


if __name__ == '__main__':
    setup_package()