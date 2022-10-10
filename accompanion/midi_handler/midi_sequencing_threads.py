# -*- coding: utf-8 -*-
import copy
import threading
import time

import numpy as np

from accompanion.config import CONFIG


class MidiInputPlayer(threading.Thread):
    def __init__(self, in_port, out_port, chords):
        threading.Thread.__init__(self)

        self.chords = chords
        self.out_port = out_port
        self.in_port = in_port

    def run(self):
        local_chords = copy.deepcopy(self.chords)
        while True:
            for In_msg in self.in_port.iter_pending():
                vel = In_msg.velocity
                if In_msg.type == "note_on":
                    for note in local_chords[0].notes:
                        note._note_on.velocity = vel
                        self.out_port.send(note._note_on)

                elif In_msg.type == "note_off":
                    for note in local_chords[0].notes:
                        self.out_port.send(note._note_off)

                    local_chords = np.delete(local_chords, 0)

            time.sleep(0.005)


class ScoreSequencer(threading.Thread):
    def __init__(self, score_or_notes, outport=None, mediator=None):

        threading.Thread.__init__(self)
        self.outport = outport

        if isinstance(score_or_notes, list):
            notes = score_or_notes
        else:
            # check for type (weird behavior...)
            notes = list(score_or_notes.notes)
        self.notes = np.array(sorted(notes, key=lambda x: x.onset))
        self.play = False
        self.mediator = mediator
        # Initial time is set to None
        self.init_time = None
        # initialize performed score onsets
        self.performed_score_onsets = [-np.inf]
        self.curr_frame = np.zeros(CONFIG["MIDI_KEYS"], dtype=np.uint8)
        self.end_of_piece = False
        self.last_performed_note = None

    def _next_notes(self, t):
        try:
            return sorted(
                [n for n in self.notes if n.p_onset >= t], key=lambda x: x.p_onset
            )
        except Exception as e:
            print("it happens here")
            print(e)

    def panic_button(self):
        """
        Stop playing and send note off messages for all
        MIDI pitches
        """
        # self.play = False
        # better use Mido's built-in panic button...
        try:
            print("Trying to note off all notes.")
            self.outport.panic()
            self.outport.reset()

        except AttributeError:
            pass
        # self.outport.reset()

    def run(self):

        # Dictionary for holding currently sounding notes
        sounding_notes = {}

        if self.init_time is None:
            # Set initial time if none given
            self.init_time = time.time()

        # set playing to true
        self.play = True

        self.curr_frame = np.zeros(CONFIG["MIDI_KEYS"], dtype=np.uint8)
        next_notes = self.notes

        while self.play:

            # current time
            c_time = time.time() - self.init_time

            # Get next note offs
            note_offs = sorted(
                [sounding_notes[p] for p in sounding_notes], key=lambda x: x.p_offset
            )

            # Send note offs
            for n_off in note_offs:
                if c_time >= n_off.p_offset:
                    # send note off message
                    self.outport.send(n_off.note_off)
                    # Remove note from sounding notes dict
                    del sounding_notes[n_off.pitch]
                    self.curr_frame[n_off.pitch - 21] = 0

            # If there are notes to send
            for n_on in next_notes:
                # Send note on messages is the note has not been
                # performed already
                if c_time >= n_on.p_onset and not n_on.already_performed:

                    if n_on.pitch in sounding_notes:
                        # send note off if the pitch is the same as an already sounding note
                        self.outport.send(sounding_notes[n_on.pitch].note_off)

                    else:
                        # add note to the list
                        sounding_notes[n_on.pitch] = n_on
                        self.curr_frame[n_on.pitch - 21] = n_on.velocity

                    # send note on
                    self.outport.send(n_on.note_on)
                    self.last_performed_note = n_on
                    if n_on.onset not in self.performed_score_onsets:
                        self.performed_score_onsets.append(n_on.onset)
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
                print("End of the piece")
                self.end_of_piece = True
                self.play = False
                self.panic_button()

    def get_midi_frame(self):
        return self.curr_frame

    def stop_playing(self):

        self.play = False
        self.panic_button()
        # self.join()
