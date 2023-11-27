from distutils.extension import Extension
from sys import platform

import numpy
import setuptools
from Cython.Build import cythonize
from setuptools import setup

# Package meta-data.
NAME = "accompanion"
DESCRIPTION = "An expressive accompaniment system"
KEYWORDS = "music alignment accompaniment"
URL = "https://github.com/CPJKU/accompanion"
EMAIL = "carloscancinochacon@gmail.com"
AUTHOR = "Carlos Cancino-Chacón, Silvan Peter, Florian Henkel, Martin Bonev"
REQUIRES_PYTHON = ">=3.7"
VERSION = "0.3.0"

include_dirs = [numpy.get_include()]

extensions = [
    Extension(
        "accompanion.mtchmkr.dtw_loop",
        ["accompanion/mtchmkr/dtw_loop.pyx"],
        include_dirs=include_dirs,
    ),
    Extension(
        "accompanion.mtchmkr.distances",
        ["accompanion/mtchmkr/distances.pyx"],
        include_dirs=include_dirs,
    ),
]

REQUIRED = [
    "python-rtmidi",
    "mido",
    "cython",
    "numpy",
    "scipy",
    "madmom",
    "partitura",
    "tqdm",
]

EXTRAS = {"fluid": ["pyfluidsynth"]}

SCRIPTS = [
    "bin/panic_button",
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    packages=setuptools.find_packages(),
    keywords=KEYWORDS,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    ext_modules=cythonize(
        extensions,
        annotate=True,
        language_level=3,
    ),  # enables generation of the html annotation file
    scripts=SCRIPTS if platform not in ["win32", "win64", "cygwin"] else None,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Other Audience",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: MusicInformationRetrieval",
    ],
)
