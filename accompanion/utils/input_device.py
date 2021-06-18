import numpy as np
import mido
import time
import threading

from mido import Message


class InputMIDI(threading.Thread):
    def __init__(self, input_port):
        threading.Thread.__init__(self)

        self.input_port = input_port
        self.midi_messages = []
        self.note_ons = []
        self.note_offs = []

    def run(self):
        for msg in self.input_port:
            print(msg)
            if msg.type == 'note_on':
                self.note_ons.append((msg, time.time()))
            elif msg.type == 'note_off':
                self.note_offs.append((msg, time.time()))


input_port = mido.open_input('USB Uno MIDI Interface')
output_port = mido.open_output('USB Uno MIDI Interface')


inmidi = InputMIDI(input_port)
on_times = []
off_times = []

note = 30
inmidi.start()
for i in range(10):
    print('sending notes')
    on_times.append(time.time())
    output_port.send(Message('note_on', note=note, velocity=48))
    time.sleep(1)
    off_times.append(time.time())
    output_port.send(Message('note_off', note=note))
    time.sleep(0.5)
# inmidi.join()
on_times = np.array(on_times)
c_on_times = np.array([t[1] for t in inmidi.note_ons])

off_times = np.array(off_times)
c_off_times = np.array([t[1] for t in inmidi.note_offs])


print(c_on_times - on_times)

print((c_on_times - on_times).mean())

print(c_off_times - off_times)
