# -*- coding: utf-8 -*-
"""
Top level of the package
"""
import platform
import pkg_resources
import importlib.util

# from . import accompanist
# from . import misc
# from . import mtchmkr
# from . import visualization

# OS: Linux, Mac or Windows
PLATFORM = platform.system()


# check if pyfluidsynth is installed
spec = importlib.util.find_spec("fluidsynth")
if spec is None:
    # Do not use a soundfont
    SOUNDFONT = None
else:
    SOUNDFONT = pkg_resources.resource_filename(
        "accompanion",
        "sound_fonts/Acoustic_Piano.sf2",
    )
