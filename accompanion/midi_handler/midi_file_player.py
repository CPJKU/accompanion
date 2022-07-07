# -*- coding: utf-8 -*-
import mido
import multiprocessing
import threading


class MidiFilePlayerThread(threading.Thread):

    def __init__(self, port_name, filename, player_class, bypass_audio=False):
        threading.Thread.__init__(self)
        self.mid = mido.MidiFile(filename)
        self.port_name = port_name
        self.midi_out = None
        self.continue_playing = True
        self.player_class = player_class
        self.bypass_audio = bypass_audio

    def run(self):
        self.open_port()
        fluidsynth_player = self.player_class()
        for msg in self.mid.play():

            if not self.continue_playing:
                break

            # is this safety necessary for fluidsynth?
            if msg.type in ('note_on', 'note_off', 'control_change'):
                self.midi_out.send(msg)

                if not self.bypass_audio:
                    fluidsynth_player.send(msg)

    def stop_playing(self):
        self.continue_playing = False
        self.join()

    def open_port(self):
        self.midi_out = mido.open_output(self.port_name)


class MidiFilePlayerProcess(multiprocessing.Process):

    def __init__(self, port_name, filename, player_class, bypass_audio=False):
        multiprocessing.Process.__init__(self)
        self.mid = mido.MidiFile(filename)
        self.port_name = port_name
        self.midi_out = None
        self.continue_playing = True
        self.player_class = player_class
        self.bypass_audio = bypass_audio

    def run(self):
        self.open_port()
        fluidsynth_player = self.player_class()
        for msg in self.mid.play():
            
            if not self.continue_playing:
                break

            # is this safety necessary for fluidsynth?
            if msg.type in ('note_on', 'note_off', 'control_change'):
                self.midi_out.send(msg)
                if not self.bypass_audio:
                    fluidsynth_player.send(msg)

    def stop_playing(self):
        self.continue_playing = False
        self.terminate()
        self.join()

    def open_port(self):
        self.midi_out = mido.open_output(self.port_name)


def get_midi_file_player(port_name, 
                         file_name, 
                         player_class, 
                         thread=False, 
                         bypass_audio=False):
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

    return file_player_type(port_name, file_name, player_class, bypass_audio)

