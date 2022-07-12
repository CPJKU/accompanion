# -*- coding: utf-8 -*-
import copy
import mido
import partitura
import threading
import time

import numpy as np

from accompanion.visualization.midi_helper import MIDI_KEYS
from scipy.interpolate import interp1d


def perf_score_map_from_match(match_path):
    """
    Parameters
    ----------
    match_path : string
        a path of a match file

    Returns
    -------
    """
    ppart, alignment, part = partitura.load_match(match_path, create_part=True)

    beat_map = part.beat_map

    perf_times = []
    score_times = []

    ppart_by_id = {note["id"]: note["note_on"] for note in ppart.notes}
    part_by_id = {note.id: beat_map(note.start.t) for note in part.notes_tied}

    for line in alignment:
        if line["label"] == "match":
            perf_times.append(ppart_by_id[line["performance_id"]])
            score_times.append(part_by_id[line["score_id"]])

    times = np.array(sorted(zip(perf_times, score_times), key=lambda x: x[0]))
    # import pdb; pdb.set_trace()
    # for x,y in zip(perf_times, score_times):
    #     print(x,y)
    perf_score_map = interp1d(
        times[:, 0], times[:, 1], fill_value="extrapolate", kind="previous"
    )

    return perf_score_map


class MODIFIEDMidiFilePlayer(threading.Thread):
    def __init__(self, out_port, filename, match_filename, virtual=False):
        threading.Thread.__init__(self)

        self.mid = mido.MidiFile(filename)
        self.out_port = out_port
        self.perf_score_map = perf_score_map_from_match(match_filename)
        self.current_s_time = 0
        self.current_vel = 64

    def run(self):
        start_time = time.time()
        # set the pedal to zero
        self.out_port.send(mido.Message(type="control_change", control=64, value=0))
        for msg in self.mid.play():
            self.out_port.send(mido.Message(type="control_change", control=64, value=0))
            self.out_port.send(msg)
            if msg.type == "note_on" and msg.velocity > 0:
                self.current_s_time = self.perf_score_map(time.time() - start_time)
                self.current_vel = msg.velocity
                # print("MIDI Player timing score / perf: ", self.current_s_time,time.time()-start_time)


class MidiInputPlayer(threading.Thread):
    def __init__(self, in_port, out_port, chords, virtual=False):
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
        self.curr_frame = np.zeros(MIDI_KEYS, dtype=np.uint8)
        self.end_of_piece = False
        self.last_performed_note = None

    def _next_notes(self, t):
        # eps = 1e-4
        return sorted(
            [n for n in self.notes if n.p_onset >= t], key=lambda x: x.p_onset
        )

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

        self.curr_frame = np.zeros(MIDI_KEYS, dtype=np.uint8)
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
                # print('note offs')
                if c_time >= n_off.p_offset:
                    # send note off message
                    # print('note off', note_offs[0].note_off)
                    self.outport.send(n_off.note_off)
                    # Remove note from sounding notes dict
                    del sounding_notes[n_off.pitch]
                    self.curr_frame[n_off.pitch - 21] = 0

            # If there are notes to send
            for n_on in next_notes:
                # print('note ons')
                # Send note on messages is the note has not been
                # performed already
                if c_time >= n_on.p_onset and not n_on.already_performed:

                    if n_on.pitch in sounding_notes:
                        # send note off if the pitch is the same as an already sounding note
                        # print('silence sounding note', sounding_notes[n_on.pitch].note_off)
                        self.outport.send(sounding_notes[n_on.pitch].note_off)

                    else:
                        # add note to the list
                        sounding_notes[n_on.pitch] = n_on
                        self.curr_frame[n_on.pitch - 21] = n_on.velocity

                    # send note on
                    # print('note on', n_on.note_on)
                    self.outport.send(n_on.note_on)
                    # print(n_on.pitch, n_on.onset, n_on.p_onset)
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
        self.join()


class MODIFIEDScoreSequencer(threading.Thread):
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
        self.performed_score_onsets = []
        self.curr_frame = np.zeros(MIDI_KEYS, dtype=np.uint8)
        self.tempo = 0.5
        self.last_s_onset = -10
        self.last_s_onset_update = 0
        self.alpha = 0.9

    def update_times(self, current_score_time, vel):
        if current_score_time != self.last_s_onset:
            c_time = time.time() - self.init_time
            for note in self.notes:
                score_reference_dist = note.onset - current_score_time
                # score_reference_dist_off = note.ffset - current_score_time
                perf_reference_dist = score_reference_dist * self.tempo
                # perf_reference_dist_off = score_reference_dist_off * self.tempo

                if score_reference_dist >= 0:
                    note.p_onset = c_time + perf_reference_dist
                    note.p_duration = 0.5 * note.duration * self.tempo
                    note.velocity = vel / 127 * 80 + 25
                if score_reference_dist >= 10:
                    break

            s_time_since_update = current_score_time - self.last_s_onset
            p_time_since_update = c_time - self.last_s_onset_update

            if self.last_s_onset != -10:
                self.tempo = self.alpha * self.tempo + (1 - self.alpha) * (
                    p_time_since_update / s_time_since_update
                )

            self.last_s_onset_update = c_time
            self.last_s_onset = current_score_time

            # print("______")
            # print("tempo", self.tempo)
            # print("tempo up", p_time_since_update)
            # print("tempo down", s_time_since_update)
            # print("last_s_onset_update", self.last_s_onset_update)
            # print("last_s_onset", self.last_s_onset)
            # print("______")

    def _next_notes(self, t):
        # eps = 1e-4
        return sorted(
            [n for n in self.notes if n.p_onset >= t], key=lambda x: x.p_onset
        )

    def panic_button(self):
        """
        Stop playing and send note off messages for all
        MIDI pitches
        """
        self.play = False
        # better use Mido's built-in panic button...
        self.outport.panic()
        self.outport.reset()

    def run(self):

        # Dictionary for holding currently sounding notes
        sounding_notes = {}

        if self.init_time is None:
            # Set initial time if none given
            self.init_time = time.time()

        # set playing to true
        self.play = True

        self.curr_frame = np.zeros(MIDI_KEYS, dtype=np.uint8)
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
                # print('note offs')
                if c_time >= n_off.p_offset:
                    # send note off message
                    # print('note off', note_offs[0].note_off)
                    self.outport.send(n_off.note_off)
                    # Remove note from sounding notes dict
                    del sounding_notes[n_off.pitch]
                    self.curr_frame[n_off.pitch - 21] = 0

            # If there are notes to send
            for n_on in next_notes:
                # print('note ons')
                # Send note on messages is the note has not been
                # performed already
                if c_time >= n_on.p_onset and not n_on.already_performed:
                    print(
                        "seinding notes",
                        c_time,
                        n_on.p_onset,
                        n_on.p_duration,
                        n_on.velocity,
                    )
                    if n_on.pitch in sounding_notes:
                        # send note off if the pitch is the same as an already sounding note
                        # print('silence sounding note', sounding_notes[n_on.pitch].note_off)
                        self.outport.send(sounding_notes[n_on.pitch].note_off)

                    else:
                        # add note to the list
                        sounding_notes[n_on.pitch] = n_on
                        self.curr_frame[n_on.pitch - 21] = 1

                    # send note on
                    # print('note on', n_on.note_on)
                    self.outport.send(n_on.note_on)
                    print(n_on)
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
                self.play = False

    def get_midi_frame(self):
        return self.curr_frame

    def stop_playing(self):
        self.play = False
        self.join()

    def export_midi_file(self):
        pass
