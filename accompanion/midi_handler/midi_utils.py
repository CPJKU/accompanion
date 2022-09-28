# -*- coding: utf-8 -*-
"""
CC: Is this class used anywhere? Otherwise delete!
"""
import threading
import mido


class VirtualMidiThroughPort(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def send(self, msg):
        self.outport.send(msg)

def midi_file_from_midi_msg(midi_msg_list, output_path):
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.Message('program_change', program=12, time=0))
    for msg in midi_msg_list:
        # tiks = mido.second2tick(time, mid.ticks_per_beat, 500000)
        track.append(mido.Message(msg.type, note=msg.note, velocity=msg.velocity, time=msg.time))
    mid.save(output_path)
    pass