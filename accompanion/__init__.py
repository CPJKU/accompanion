# -*- coding: utf-8 -*-
"""
Top level of the package
"""
import importlib.resources
import importlib.util
import platform

# OS: Linux, Mac or Windows
PLATFORM = platform.system()

SOUNDFONT = None

# check if pyfluidsynth is installed
spec = importlib.util.find_spec("fluidsynth")
HAS_FLUIDSYNTH = spec is not None
if HAS_FLUIDSYNTH:
    # `pkg_resources` was removed in setuptools 81
    SOUNDFONT = str(
        importlib.resources.files("accompanion") / "sound_fonts" / "Acoustic_Piano.sf2"
    )
