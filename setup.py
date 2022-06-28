import numpy
import setuptools
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='accompanion',
    version='0.3.0',
    description='accompanion',
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: MusicInformationRetrieval",
    ],
    author='Carlos Cancino-Chacón, Silvan Peter, Florian Henkel, Martin Bonev',
    ext_modules=cythonize(
        ['mtchmkr/dtw_loop.pyx'],
        annotate=True),                 # enables generation of the html annotation file
    include_dirs=[numpy.get_include()],
)
