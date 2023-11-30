# -*- coding: utf-8 -*-
"""
Sequencer for MIDI playback.

TODO
----
* Document module
* Would this work on Windows? Very unlikely... :(
"""
import warnings

from accompanion import HAS_FLUIDSYNTH, PLATFORM

if PLATFORM == "Linux":
    MIDI_DRIVER = "alsa"
elif PLATFORM == "Darwin":
    MIDI_DRIVER = "coreaudio"
elif PLATFORM == "Windows":
    # raise ValueError("Windows is not supported yet...")
    print("No Fluidsynth Windows support yet...")
    MIDI_DRIVER = None


if PLATFORM in ("Linux", "Darwin") and HAS_FLUIDSYNTH:
    import fluidsynth

    from accompanion import SOUNDFONT

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

else:

    class FluidsynthPlayer(object):
        def __init__(self):
            self.name = "FluidsynthPlayer dummy"
            warnings.warn(
                "Fluidsynth Player is currently not supported under "
                "Windows, use a MIDI output port!"
            )

        def send(self, message):
            pass

        def delete(self):
            pass

        def panic(self):
            pass
