from sys import platform

import setuptools
from setuptools import setup

# Package meta-data.
NAME = "accompanion"
DESCRIPTION = "An expressive accompaniment system"
KEYWORDS = "music alignment accompaniment"
URL = "https://github.com/CPJKU/accompanion"
EMAIL = "carloscancinochacon@gmail.com"
AUTHOR = "Carlos Cancino-Chacón, Silvan Peter, Florian Henkel, Martin Bonev"
# matchmaker, which provides the score followers, supports 3.10 - 3.12.
REQUIRES_PYTHON = ">=3.10,<3.13"
VERSION = "0.3.0"

# Branch of pymatchmaker/matchmaker providing the score following methods.
# TODO: switch to a released version of `pymatchmaker` once this branch has
# been merged into `develop`.
MATCHMAKER_BRANCH = "feature/clean_method_registration"

REQUIRED = [
    "mido",
    # mido's MIDI I/O backend. Imported by mido at runtime, not by name here.
    "python-rtmidi",
    "numpy>=1.26.3,<2",
    # basismixer.performance_codec calls scipy.misc.derivative, which SciPy
    # removed in 1.15.
    "scipy>=1.11.4,<1.15",
    "partitura>=1.9.0",
    "pyyaml",
    # basismixer.utils.music imports torch, though basismixer does not declare
    # it. Install the CPU build first (see environment.yml) to avoid pulling
    # the much larger CUDA wheels.
    "torch",
    # The ACCompanion still needs basismixer for `get_performance_codec`, which
    # partitura does not provide.
    "basismixer @ git+https://github.com/OFAI/basismixer.git",
    # Score followers.
    "pymatchmaker @ git+https://github.com/pymatchmaker/matchmaker.git"
    f"@{MATCHMAKER_BRANCH}",
    # Used by bin/app.py, the GUI launcher. Pinned because PySimpleGUI 5
    # changed to a licensed distribution model.
    "PySimpleGUI==4.60.5.1",
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
