# -*- coding: utf-8 -*-
"""
Sequencer for MIDI playback.

TODO
----
* Document module
* Would this work on Windows? Very unlikely... :(
"""
import os
import platform

import warnings

# Put this as global variables somewhere else
PLATFORM = platform.system()

if PLATFORM == "Linux":
    MIDI_DRIVER = "alsa"
elif PLATFORM == "Darwin":
    MIDI_DRIVER = "coreaudio"
elif PLATFORM == "Windows":
    # raise ValueError("Windows is not supported yet...")
    print("No Fluidsynth Windows support...")
    MIDI_DRIVER = None

# for some reason, Mac does not like relative paths...
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(CURRENT_DIR)
SOUNDFONT = os.path.join(os.path.dirname(CURRENT_DIR), "sound_fonts",
                         "Acoustic_Piano.sf2")
if PLATFORM != "Windows":
    import fluidsynth

class FluidsynthPlayer(object):
    """
    Play a MIDI input using Fluidsynth

    TODO
    ----
    * this could be setup as a port following the mido Port API
      and thus simplfy Score Sequencer. (add panic, send...).
    """
    def __init__(self, soundfont=SOUNDFONT):
        self.fs = fluidsynth.Synth()
        self.fs.start(driver=MIDI_DRIVER)
        sfid = self.fs.sfload(soundfont)
        self.fs.program_select(0, sfid, 0, 0)

    def send(self, msg):
        if msg is not None:
            if msg.type == "note_on":
                self.fs.noteon(0, msg.note, msg.velocity)

            elif msg.type == "note_off":
                self.fs.noteoff(0, msg.note)

            elif msg.type == "control_change":
                self.fs.cc(0, msg.control, msg.value)

    def delete(self):
        self.fs.delete()

    def panic(self):
        for mp in range(128):
            self.fs.noteoff(0, mp)
            


class FluidsynthPlayerWindows(object):
    def __init__(self):
        self.name = "FluidsynthPlayer dummy"

    def send(self, message):
        warnings.warn(
            f"Fluidsynth Player is currently not supported under "
            "windows, use a MIDI output port!"
        )

    def panic(self):
        pass
