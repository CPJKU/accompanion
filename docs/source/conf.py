# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import re
import sys

# Make the package importable without installing it.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def _setup_value(name):
    """Read a constant (e.g. VERSION) from setup.py, without importing it."""
    with open(os.path.join(ROOT, "setup.py"), encoding="utf-8") as f:
        match = re.search(rf'^{name}\s*=\s*"([^"]*)"', f.read(), re.M)
    return match.group(1) if match else ""


# -- Project information -----------------------------------------------------

project = "ACCompanion"
author = _setup_value("AUTHOR")
copyright = "2017-2026, Institute of Computational Perception, JKU Linz"
release = _setup_value("VERSION")
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- API reference -------------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True

# Dependencies that are heavy, compiled, need hardware or are only installable
# from git. They are mocked so the API reference builds on Read the Docs
# without installing the ACCompanion's full environment.
autodoc_mock_imports = [
    "basismixer",
    "fluidsynth",
    "hiddenmarkov",
    "matchmaker",
    "matplotlib",
    "mido",
    "misc",
    "partitura",
    "PySimpleGUI",
    "rtmidi",
    "torch",
    "yaml",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "partitura": ("https://partitura.readthedocs.io/en/latest/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"ACCompanion {release}"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/CPJKU/accompanion/",
    "source_branch": "main",
    "source_directory": "docs/source/",
}
