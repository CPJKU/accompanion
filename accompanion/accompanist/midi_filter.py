import mido
import numpy as np
import time
import threading
from queue import Queue, Empty
from threading import Lock


class MidiThread(threading.Thread):

    def __init__(self, midi_path, midi_out, filter_list, lock):
        threading.Thread.__init__(self)

        self.midi_outport = midi_out
        self.path_midi = midi_path
        self.midi = None
        self.load_midi(self.path_midi)
        # self.queue = queue
        self.filter_list = filter_list
        self.lock = lock

    def load_midi(self, path_midi):
        self.midi = mido.MidiFile(path_midi)

    def run(self):
        for msg in self.midi.play():
            play_msg = msg
            # time.sleep(msg.time)

            if msg.type == 'note_on' or msg.type == 'note_off':
                if msg.type == 'note_on' and msg.velocity > 0:
                    # self.queue.put((play_msg.note, time.time()))
                    with self.lock:
                        self.filter_list.append(play_msg.note)
                    # play_msg.velocity = 33
                self.midi_outport.send(play_msg)

        return 0


class MIDIList(object):

    def __init__(self):
        self.notes = []

    def append(self, notes):
        self.notes.append((notes, time.time()))

    def __len__(self):
        return len(self.notes)

    def delete(self, idx):
        del self.notes[idx]

    def __getitem__(self, idx):
        return self.notes[idx]

    def __str__(self):
        return str(self.notes)

    def pop(self, note):
        '''
        Pop the element of the list
        '''


class PlayMidiThread(threading.Thread):
    def __init__(self, midi_in, filter_list, lock):
        threading.Thread.__init__(self)
        self.midi_inport = midi_in
        # self.queue = queue
        self.filter_list = filter_list
        self.messages = MIDIList()
        self.lock = lock

    def run(self):
        for msg in self.midi_inport:

            if msg.type == 'note_on':
                c_time = time.time()
                # self.messages.append(msg.note)
                if len(self.filter_list) > 0:
                    with self.lock:
                        q_p, q_t = self.filter_list[-1]
                        if msg.note == q_p and abs(c_time - q_t) <= 0.22:
                            print('same note')

                            self.filter_list.delete(-1)
                        else:
                            self.messages.append(msg)

                else:
                    self.messages.append(msg)


if __name__ == '__main__':

    fn = './Test_score-Part_2.mid'
    outport = mido.open_output('USB Uno MIDI Interface')
    inport = mido.open_input('USB Uno MIDI Interface')
    acc_queue = Queue()
    filter_list = MIDIList()
    acc_list = []
    lock = Lock()
    mt = MidiThread(fn, outport, filter_list=filter_list, lock=lock)
    pt = PlayMidiThread(inport, filter_list=filter_list, lock=lock)
    pt.start()
    mt.start()
