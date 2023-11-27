# -*- coding: utf-8 -*-
import multiprocessing
import threading

import mido


class MidiFilePlayerThread(threading.Thread):
    def __init__(self, port, filename, player_class, bypass_audio=False):
        threading.Thread.__init__(self)
        self.mid = mido.MidiFile(filename)
        self.midi_out = port
        self.continue_playing = True
        self.player_class = player_class
        self.bypass_audio = bypass_audio

    def run(self):
        fluidsynth_player = self.player_class()
        for msg in self.mid.play():

            if not self.continue_playing:
                break

            # is this safety necessary for fluidsynth?
            if msg.type in ("note_on", "note_off", "control_change"):
                self.midi_out.send(msg)

                if not self.bypass_audio:
                    fluidsynth_player.send(msg)

    def stop_playing(self):
        self.continue_playing = False


class MidiFilePlayerProcess(multiprocessing.Process):
    def __init__(self, port, filename, player_class, bypass_audio=False):
        multiprocessing.Process.__init__(self)
        self.mid = mido.MidiFile(filename)
        self.midi_out = port
        self.continue_playing = True
        self.player_class = player_class
        self.bypass_audio = bypass_audio

    def run(self):

        fluidsynth_player = self.player_class()
        for msg in self.mid.play():

            if not self.continue_playing:
                break

            # is this safety necessary for fluidsynth?
            if msg.type in ("note_on", "note_off", "control_change"):
                self.midi_out.send(msg)
                if not self.bypass_audio:
                    fluidsynth_player.send(msg)

    def stop_playing(self):
        self.continue_playing = False
        self.terminate()
        self.join()


def get_midi_file_player(
    port,
    file_name,
    player_class,
    thread=False,
    bypass_audio=False,
):
    # import pdb
    # pdb.set_trace()
    # print(f'dodo {filename}')

    if thread:
        file_player_type = MidiFilePlayerThread
    else:
        # from multiprocessing import Manager
        # from multiprocessing.managers import BaseManager
        # BaseManager.register('FluidsynthPlayer', player_class)
        # manager = BaseManager()
        # manager.start()
        # inst = manager.FluidsynthPlayer()
        file_player_type = MidiFilePlayerProcess

    return file_player_type(port, file_name, player_class, bypass_audio)
