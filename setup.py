import numpy
import setuptools
from setuptools import setup
from Cython.Build import cythonize

# Package meta-data.
NAME = 'accompanion'
DESCRIPTION = 'An expressive accompaniment system'
KEYWORDS = 'music alignment midi audio'
URL = 'https://github.com/CPJKU/accompanion'
EMAIL = 'carloscancinochacon@gmail.com'
AUTHOR = 'Carlos Cancino-Chacón, Silvan Peter, Florian Henkel, Martin Bonev'
REQUIRES_PYTHON = '>=3.7'
VERSION = '0.3.0'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=setuptools.find_packages(),
    keywords=KEYWORDS,
    author_email=EMAIL,
    author=AUTHOR,
    url=URL,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: MusicInformationRetrieval",
    ],
    ext_modules=cythonize(
        ['mtchmkr/dtw_loop.pyx'],
        annotate=True),                 # enables generation of the html annotation file
    include_dirs=[numpy.get_include()],
)
