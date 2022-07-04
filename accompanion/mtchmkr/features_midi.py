# -*- coding: utf-8 -*-
"""
Features from symbolic files

TODO
----
* Set a more usable format for `mido.Message` objects from input
"""
import numpy as np

# import mido
# import partitura


class Pitch_IOI_Processor(object):
    def __init__(self, piano_range=False):
        self.prev_time = 0
        self.piano_range = piano_range

        self.pitch_bias = 21 if piano_range else 0

    def __call__(self, frame, kwargs={}):
        data, f_time = frame
        pitch_obs = []
        self.ref_time = f_time
        ioi_obs = f_time - self.ref_time
        self.prev_time = f_time

        for msg, t in data:
            if (
                getattr(msg, "type", "other") == "note_on"
                and getattr(msg, "velocity", 0) > 0
            ):
                pitch_obs.append(msg.note)

        return (np.array(pitch_obs) - self.pitch_bias, ioi_obs), {}


class PianoRollProcessor(object):
    def __init__(self, use_velocity=False, piano_range=False, dtype=float):
        self.active_notes = dict()
        self.piano_roll_slices = []
        self.use_velocity = use_velocity
        self.piano_range = piano_range
        self.dtype = dtype

    def __call__(self, frame, kwargs={}):
        # initialize piano roll
        piano_roll_slice = np.zeros(128, dtype=self.dtype)
        data, f_time = frame
        for msg, m_time in data:
            # TODO: update with new format, if Mido Messages
            # messages are substituted for something else
            if msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and msg.velocity > 0:
                    self.active_notes[msg.note] = (msg.velocity, m_time)
                else:
                    try:
                        del self.active_notes[msg.note]
                    except KeyError:
                        pass

        for note, (vel, m_time) in self.active_notes.items():
            if self.use_velocity:
                piano_roll_slice[note] = vel
            else:
                piano_roll_slice[note] = 1

        if self.piano_range:
            piano_roll_slice = piano_roll_slice[21:109]
        self.piano_roll_slices.append(piano_roll_slice)

        return piano_roll_slice, {}

    def reset(self):
        self.piano_roll_slices = []
        self.active_notes = dict()


class CumSumPianoRollProcessor(object):
    def __init__(self, use_velocity=False, piano_range=False, dtype=float):
        self.active_notes = dict()
        self.piano_roll_slices = []
        self.use_velocity = use_velocity
        self.piano_range = piano_range
        self.dtype = dtype

    def __call__(self, frame, kwargs={}):
        # initialize piano roll
        piano_roll_slice = np.zeros(128, dtype=self.dtype)
        data, f_time = frame
        for msg, m_time in data:
            # TODO: update with new format, if Mido Messages
            # messages are substituted for something else
            if msg.type in ("note_on", "note_off"):
                if msg.type == "note_on" and msg.velocity > 0:
                    self.active_notes[msg.note] = (msg.velocity, m_time)
                else:
                    try:
                        del self.active_notes[msg.note]
                    except KeyError:
                        pass

        for note, (vel, m_time) in self.active_notes.items():
            if self.use_velocity:
                piano_roll_slice[note] = vel
            else:
                piano_roll_slice[note] = 1

        if self.piano_range:
            piano_roll_slice = piano_roll_slice[21:109]
        self.piano_roll_slices.append(piano_roll_slice)

        return piano_roll_slice, {}

    def reset(self):
        self.piano_roll_slices = []
        self.active_notes = dict()
