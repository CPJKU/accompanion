# -*- coding: utf-8 -*-
"""
MIDI utilities
"""
import threading
import os

import mido

from os.path import dirname, realpath

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
acc_pack_dir = dirname(dir_of_file)
accdir = dirname(acc_pack_dir)


class VirtualMidiThroughPort(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def send(self, msg):
        self.outport.send(msg)


OUTPUT_MIDI_FOLDER = os.path.join(accdir, "recorded_midi")
if not os.path.exists(OUTPUT_MIDI_FOLDER):
    os.makedirs(OUTPUT_MIDI_FOLDER)


def midi_file_from_midi_msg(midi_msg_list, output_path):
    """Save a midi file, given a sequence of midi messages with a absolute time stamp.

    Args:
        midi_msg_list list: the list of pairs (midi_msg, time_stamp)
        output_path str: the path of the output midi file
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # track.append(mido.Message('program_change', program=12, time=0))
    starting_time = midi_msg_list[0][1]
    last_message_time = starting_time
    for msg, abs_time in midi_msg_list:
        # TODO: Add pedal messages
        if msg.type in ("note_on", "note_off"):
            delta_time = abs_time - last_message_time
            last_message_time = abs_time
            ticks = int(mido.second2tick(delta_time, mid.ticks_per_beat, 500000))
            track.append(
                mido.Message(
                    msg.type,
                    note=msg.note,
                    velocity=msg.velocity,
                    time=ticks,
                )
            )
    mid.save(output_path)
