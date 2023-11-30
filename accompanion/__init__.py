# -*- coding: utf-8 -*-
"""
Top level of the package
"""
import importlib.util
import platform

import pkg_resources

# OS: Linux, Mac or Windows
PLATFORM = platform.system()

SOUNDFONT = None

# check if pyfluidsynth is installed
spec = importlib.util.find_spec("fluidsynth")
HAS_FLUIDSYNTH = spec is not None
if HAS_FLUIDSYNTH:
    SOUNDFONT = pkg_resources.resource_filename(
        "accompanion",
        "sound_fonts/Acoustic_Piano.sf2",
    )
