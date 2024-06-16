# -*- coding: utf-8 -*-


from setuptools import setup, find_packages

from qosa import __version__ as VERSION


DESCRIPTION = 'A set of python modules for computing the qosa indices with the Random Forest method'
DISTNAME = 'qosa-indices'
LICENSE = 'GPL V3'
URL = 'https://gitlab.com/qosa_index/qosa'

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

with open('requirements.txt') as f:
    REQUIRED = f.read().splitlines()

# to preventg error message after loading
zip_safe=False

setup(
    # Name of the library, such as it will appear on PyPi
    name = DISTNAME,

    # Version of the code
    version = VERSION,

    # A short description of the package
    description = DESCRIPTION,

    # A long description will be printed to present the library.
    # Generally, we dump the README here
    long_description = LONG_DESCRIPTION,

    # An URL that points to the official page of the library
    url = URL,

    author = 'Kévin Elie-Dit-Cosaque',
    author_email = 'edckev@gmail.com',
    license = LICENSE,
    keywords = ['sklearn', 'randomforest quantile'],

    # Lists the packages to be inserted in the distribution rather than doing 
    # it manually, we use the find_packages() function of setuptools which will
    # search all python packages recursively in the current folder.
    packages = find_packages(),

    # List of dependencies for the library
    install_requires = REQUIRED

)