# Installation

## Prerequisites

* [Miniforge](https://github.com/conda-forge/miniforge) or another conda
  distribution. Check with `conda --version`.
* [Git](https://git-scm.com/downloads). Check with `git --version`.
* On Windows only, a C compiler: the "Desktop development with C++" workload
  of the Visual Studio Build Tools. The score following backend
  ([matchmaker](https://github.com/pymatchmaker/matchmaker)) has Cython
  extensions, and PyPI has prebuilt wheels for it on Linux and macOS only.
  On Windows, install it from its release tag before `pip install -e .`:
  `pip install "pymatchmaker @ git+https://github.com/pymatchmaker/matchmaker.git@v0.4.0"`
* On Windows, delete the `- fluidsynth` line from `environment.yml` before
  creating the environment: conda-forge has no FluidSynth for Windows, and the
  ACCompanion only uses it on Linux and macOS.

## Installing

```shell
git clone https://github.com/CPJKU/accompanion.git
cd accompanion
git submodule update --init          # the accompanion_pieces submodule
conda env create -f environment.yml
conda activate accompanion
pip install -e .
```

`environment.yml` provides the interpreter and the handful of packages that
need conda: the ones with a compiled or system component, and the ones whose
version has to be held back. Everything else is declared in `setup.py` and
installed by `pip install -e .`, including
[basismixer](https://github.com/OFAI/basismixer), which comes from git.

Expect the install to take a few minutes and about 2.5 GB, most of which is
PyTorch and the math libraries it pulls in.

## Checking the installation

```shell
python ./bin/test_followers.py --list
python ./bin/test_followers.py --piece bach_menuett
```

The first command lists the score followers your installation provides; the
second runs all of them over a rendered performance and prints how closely each
one tracked it. Neither needs MIDI hardware or audio. The first run of anything
that imports partitura also downloads a ~35 MB soundfont over FTP, which some
networks block; it is cached afterwards.

## Why the versions are pinned

These constraints are not obvious from the errors they produce when violated,
so `environment.yml` pins them with a comment:

| Pin | Reason |
| --- | --- |
| `python=3.11` | The version the ACCompanion is tested on. matchmaker supports 3.10 - 3.13. |
| `scipy<1.15` | basismixer calls `scipy.misc.derivative`, removed in SciPy 1.15. |
| `setuptools<81` | basismixer imports `pkg_resources`, removed in setuptools 81. |
| `pytorch-cpu` | basismixer imports torch. Installing the CPU build from conda-forge keeps pip from pulling the much larger CUDA wheels. |

## Upgrading from an older ACCompanion

Environments created before the move to matchmaker run Python 3.9, which
matchmaker does not support. Recreate rather than update:

```shell
conda env remove -n accompanion
conda env create -f environment.yml
conda activate accompanion
pip install -e .
```
