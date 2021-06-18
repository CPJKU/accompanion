# -*- coding: utf-8 -*-
"""
Sequencer for MIDI playback.

TODO
----
* Send midi messages to a filter in the score follower (for CEUS)
* Document module
"""
import numpy as np
import mido
import threading
import time

from mido import Message

try:
    from score import Note, Chord, Score
except ModuleNotFoundError:
    from .score import Note, Chord, Score


class ScoreSequencer(threading.Thread):

    def __init__(self, notes, outport, mediator=None):

        threading.Thread.__init__(self)

        self.midi_outport = outport
        self.notes = np.array(sorted(notes, key=lambda x: x.onset))
        self.play = False
        self.mediator = mediator
        # Initial time is set to None
        self.init_time = None

    def _next_notes(self, t):

        # eps = 1e-4
        return sorted([n for n in self.notes if n.p_onset >= t],
                      key=lambda x: x.p_onset)

    def panic_button(self):
        """
        Stop playing and send note off messages for all
        MIDI pitches
        """

        self.play = False
        for i in range(128):
            # Send Note offs for all MIDI pitches
            self.midi_outport.send(
                Message('note_off',
                        note=i,
                        velocity=64))

    def run(self):

        # Dictionary for holding currently sounding notes
        sounding_notes = {}

        if self.init_time is None:
            # Set initial time if none given
            self.init_time = time.time()

        # set playing to true
        self.play = True

        next_notes = self.notes
        while self.play:

            # current time
            c_time = time.time() - self.init_time

            # Get next note offs
            note_offs = sorted([sounding_notes[p] for p in sounding_notes],
                               key=lambda x: x.p_offset)

            # Send note offs
            for n_off in note_offs:
                # print('note offs')
                if (c_time >= n_off.p_offset):
                    # send note off message
                    # print('note off', note_offs[0].note_off)
                    self.midi_outport.send(n_off.note_off)
                    # Remove note from sounding notes dict
                    del sounding_notes[n_off.pitch]

            # If there are notes to send
            for n_on in next_notes:
                # print('note ons')
                # Send note on messages is the note has not been
                # performed already
                if c_time >= n_on.p_onset and not n_on.already_performed:

                    if n_on.pitch in sounding_notes:
                        # send note off if the pitch is the same as an already sounding note
                        # print('silence sounding note', sounding_notes[n_on.pitch].note_off)
                        self.midi_outport.send(sounding_notes[n_on.pitch].note_off)

                    else:
                        # add note to the list
                        sounding_notes[n_on.pitch] = n_on

                    # send note on
                    # print('note on', n_on.note_on)
                    self.midi_outport.send(n_on.note_on)
                    # set note to already performed
                    n_on.already_performed = True

                    # Add notes to the mediator, so to avoid them
                    # being caught by the score follower
                    if self.mediator is not None:
                        self.mediator.filter_append_pitch(n_on.pitch)

            # Get next notes to be performed
            next_notes = self._next_notes(c_time)

            time.sleep(1e-6)

            # Stop playing if there are no more notes to send
            if len(next_notes) == 0 and len(sounding_notes) == 0:
                print('End of the piece')
                self.play = False


def test_sequencer1():
    notes = [Note(60 + i, i, 1, i, 1, 45) for i in range(13)]

    outport = mido.open_output('USB Uno MIDI Interface')

    seq = ScoreSequencer(notes, outport)

    seq.start()

    try:
        while True:
            for i, n in enumerate(notes[1:]):
                n.p_onset = np.clip(i + 1 + 0.2 * np.random.randn(), i, i + 2)
                n.p_duration = 0.5 + np.random.rand()
                n.velocity = np.random.randint(30, 45)
                time.sleep(1e-3)
    except KeyboardInterrupt:
        seq.join()


if __name__ == '__main__':

    import partitura

    fn = '/Users/aae/Repos/hierarchical_tempo_analysis/datasets/vienna_4x22_corrected/Chopin_op10_no3_p03.match'
    mf = partitura.matchfile.match_to_notearray(fn, expand_grace_notes='d')

    notes = [Note(n['pitch'], n['onset'], n['duration'], n['p_onset'], n['p_duration'], n['velocity'] * .9) for n in mf[:100]]

    outport = mido.open_output('USB Uno MIDI Interface')

    seq = ScoreSequencer(notes, outport)

    seq.start()
