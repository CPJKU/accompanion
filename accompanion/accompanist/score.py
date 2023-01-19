# -*- coding: utf-8 -*-
"""
Objects for representing score information
"""
from typing import Iterable
import numpy as np

# from matchmaker.io.symbolic import load_score
from partitura import load_score
from mido import Message
from partitura.score import Part
from partitura.performance import PerformedPart

from partitura.utils.music import performance_from_part


class Note(object):
    """
    Class for representing notes
    """

    def __init__(
        self,
        pitch,
        onset,
        duration,
        p_onset=None,
        p_duration=None,
        velocity=64,
        id=None,
        channel=0,
    ):

        self.pitch = pitch
        self.onset = onset
        self.duration = duration
        self.p_onset = p_onset
        self.p_duration = p_duration
        self.id = id
        self.already_performed = False
        self.velocity = velocity
        self._note_on = Message(
            type="note_on",
            velocity=self.velocity,
            note=self.pitch,
            time=self.p_onset if self.p_onset is not None else 0,
            channel=channel,
        )

        self._note_off = Message(
            type="note_off",
            velocity=0,  # self.velocity,
            note=self.pitch,
            time=self.p_offset,
            channel=channel,
        )

    def __string__(self):
        out_string = f"Note({self.pitch}, {self.onset}, {self.p_onset})"
        return out_string

    @property
    def p_onset(self):
        return self._p_onset

    @p_onset.setter
    def p_onset(self, onset):
        self._p_onset = onset

    @property
    def note_on(self):
        self._note_on.velocity = self.velocity
        self._note_on.time = self.p_onset
        return self._note_on

    @property
    def note_off(self):
        self._note_off.velocity = self.velocity
        self._note_off.time = self.p_offset
        return self._note_off

    @property
    def p_offset(self):
        return self.p_onset + self.p_duration

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = int(velocity)


class Chord(object):
    """
    Class for representing Score onsets or "chords".
    """

    def __init__(self, notes):

        if not isinstance(notes, Iterable):
            notes = [notes]
        assert all([n.onset == notes[0].onset for n in notes])

        self.notes = notes
        self.num_notes = len(notes)

        self.onset = self.notes[0].onset
        self.pitch = np.array([n.pitch for n in self.notes])
        self.duration = np.array([n.duration for n in self.notes])

    def __getitem__(self, index):
        return self.notes[index]

    def __len__(self):
        return self.num_notes

    @property
    def p_onset(self):

        if any([n.p_onset is None for n in self.notes]):
            return None
        else:
            return np.mean([n.p_onset for n in self.notes])

    @p_onset.setter
    def p_onset(self, p_onset):
        if isinstance(p_onset, (float, int)):
            for n in self.notes:
                n.p_onset = p_onset
        else:
            # Assume that self.notes and p_onset
            # have the same length (this makes a little bit
            # faster)
            for n, po in zip(self.notes, p_onset):
                n.p_onset = po
        # do not check length?
        # elif len(p_onset) == self.num_notes:
        #     for n, po in zip(self.notes, p_onset):
        #         n.p_onset = po

    @property
    def p_duration(self):
        if any([n.p_duration is None for n in self.notes]):
            return None
        else:
            return np.mean([n.p_duration for n in self.notes])

    @p_duration.setter
    def p_duration(self, p_duration):
        if isinstance(p_duration, (float, int)):
            for n in self.notes:
                n.p_duration = p_duration
        # elif len(p_duration) == self.num_notes:
        else:
            for n, po in zip(self.notes, p_duration):
                n.p_duration = po

    @property
    def velocity(self):
        if any([n.velocity is None for n in self.notes]):
            return None
        else:
            return np.max([n.velocity for n in self.notes])

    @velocity.setter
    def velocity(self, velocity):
        if isinstance(velocity, (float, int)):
            for n in self.notes:
                n.velocity = velocity
        # elif len(velocity) == self.num_notes:
        else:
            for n, po in zip(self.notes, velocity):
                n.velocity = po


class Score(object):
    def __init__(
        self,
        notes,
        time_signature_map=None,
        access_mode="indexwise",
        note_array=None,
    ):

        # TODO: Seconday sort by pitch
        self.notes = np.array(sorted(notes, key=lambda x: x.pitch))
        self.time_signature_map = time_signature_map

        self.access_mode = access_mode
        onsets = np.array([n.onset for n in self.notes])

        # Unique score positions
        self.unique_onsets = np.unique(onsets)
        self.unique_onsets.sort()
        self.min_onset = self.unique_onsets[0]
        self.max_onset = self.unique_onsets[-1]

        # indices of the notes belonging to each
        self.unique_onset_idxs = [np.where(onsets == u) for u in self.unique_onsets]

        self.chords = np.empty(len(self.unique_onset_idxs), dtype=object)
        # Very weird numpy behavior...
        # See https://stackoverflow.com/a/72036793
        self.chords[:] = [Chord(self.notes[ui]) for ui in self.unique_onset_idxs]
        # self.chords = np.array(
        #     [Chord(self.notes[ui]) for ui in self.unique_onset_idxs], dtype=object
        # )

        # assert(all([isinstance(c, Chord) for c in self.chords]))

        self.chord_dict = dict(
            [(u, c) for u, c in zip(self.unique_onsets, self.chords)]
        )

        if note_array is None:
            self.note_array_from_notes()
        else:
            self.note_array = note_array

    def note_array_from_notes(self) -> None:
        note_array = np.zeros(
            len(self.notes),
            dtype=[
                ("pitch", "i4"),
                ("onset_beat", "f4"),
                ("duration_beat", "f4"),
                ("id", "U256"),
            ],
        )

        for i, note in enumerate(self.notes):
            note_array["pitch"][i] = note.pitch
            note_array["onset_beat"][i] = note.onset
            note_array["duration_beat"][i] = note.duration
            note_array["id"][i] = note.id

        self.note_array = note_array


    @property
    def access_mode(self):
        return self._access_mode

    @access_mode.setter
    def access_mode(self, access_mode):
        if access_mode not in ("indexwise", "timewise"):
            raise ValueError(
                '`access_mode` should be "indexwise" or "timewise". '
                "Given {0}".format(access_mode)
            )
        self._access_mode = access_mode

        if self._access_mode == "indexwise":
            self._getitem_ = self.getitem_indexwise
        elif self.access_mode == "timewise":
            self._getitem_ = self.getitem_timewise

    def getitem_indexwise(self, index):
        return self.chords[index]

    def getitem_timewise(self, index):
        return self.chord_dict[index]

    def __getitem__(self, index):
        return self._getitem_(index)

    # def __getitem__(self, index):
    #     if self.access_mode == 'indexwise':
    #         return self.chords[index]
    #     elif self.access_mode == 'timewise':
    #         return self.chord_dict[index]

    def __len__(self):
        return len(self.unique_onsets)

    def export_midi(self, out_fn):
        # create note_array
        note_array = np.zeros(
            len(self.notes),
            dtype=[
                ("pitch", "i4"),
                ("onset_sec", "f4"),
                ("duration_sec", "f4"),
                ("velocity", "i4"),
            ],
        )

        for i, note in enumerate(self.notes):

            note_array["pitch"][i] = note.pitch
            note_array["onset_sec"][i] = note.p_onset
            note_array["duration_sec"][i] = note.p_duration
            note_array["velocity"][i] = note.velocity

        ppart = partitura.performance.PerformedPart.from_note_array(note_array)

        partitura.save_performance_midi(ppart, out_fn)


class AccompanimentScore(Score):
    def __init__(
        self,
        notes,
        solo_score,
        mode="iter_solo",
        velocity_trend=None,
        velocity_dev=None,
        log_bpr=None,
        timing=None,
        log_articulation=None,
        note_array=None,
    ):
        
        # print(type(solo_score)) # partitura.score.Score
        assert isinstance(solo_score, Score)

        super().__init__(
            notes=notes,
            time_signature_map=solo_score.time_signature_map,
            access_mode="indexwise",
            note_array=note_array,
        )

        self.ssc = solo_score

        self.mode = mode

        self.solo_score_dict = dict()

        self.velocity_trend = dict()
        self.velocity_dev = dict()
        self.log_bpr = dict()
        self.timing = dict()
        self.log_articulation = dict()

        # Move this somewhere else!!!
        # Deadpan performance
        if velocity_trend is None:
            velocity_trend = np.ones(len(self.notes))
        if velocity_dev is None:
            velocity_dev = np.zeros(len(self.notes))
        if log_bpr is None:
            log_bpr = np.zeros(len(self.notes))
        if timing is None:
            timing = np.zeros(len(self.notes))
            # hack
            # timing = 0.1 * (np.random.rand(len(self.notes)) - 0.5)
        if log_articulation is None:
            log_articulation = np.zeros(len(self.notes))

        for chord, uix in zip(self.chords, self.unique_onset_idxs):
            self.velocity_trend[chord] = np.mean(velocity_trend[uix])
            self.velocity_dev[chord] = velocity_dev[uix]
            self.log_bpr[chord] = np.mean(log_bpr[uix])
            self.timing[chord] = timing[uix]
            self.log_articulation[chord] = log_articulation[uix]

        for i, on in enumerate(self.ssc.unique_onsets):

            acc_idx = np.where(self.unique_onsets >= on)[0]

            if len(acc_idx) > 0:
                acc_idx = np.min(acc_idx)
                next_acc_onsets = self.unique_onsets[acc_idx:]
                ioi_init = next_acc_onsets.min() - on

                next_iois = np.r_[ioi_init, np.diff(next_acc_onsets)]
                # next_onsets = np.cumsum(next_iois)

                # This information seems to be redundant...
                # self.next_onsets[on] = (self.chords[acc_idx:], next_iois)

                self.solo_score_dict[on] = (
                    self.chords[acc_idx:],
                    next_iois,
                    next_acc_onsets,
                    acc_idx,
                    i,
                )
            else:
                # self.next_onsets[on] = (None, None)
                self.solo_score_dict[on] = (None, None, None, None, i)


def part_to_score(fn_spart_or_ppart, bpm=100, velocity=64):
    """
    Get a `Score` instance from a partitura `Part` or `PerformedPart`

    Parameters
    ----------
    fn_spart_or_ppart : filename, Part of PerformedPart
        Filename or partitura Part
    bpm : float
        Beats per minute to generate the performance (this is ignored if
        the input is a `PerformedPart`
    velocity : int
        Velocity to generate the performance (this is ignored if the input
        is a `PerformedPart`

    Returns
    -------
    score : Score
        A `Score` object.
    """
    
    # print(type(fn_spart_or_ppart)) # partitura.score.Score
    # print(fn_spart_or_ppart) # <partitura.score.Score object at 0x7f8aaac61990>
    # print(isinstance(fn_spart_or_ppart, Score)) # False
    
    if isinstance(fn_spart_or_ppart, str):
        part = load_score(fn_spart_or_ppart)
    # elif isinstance(fn_spart_or_ppart, (Part, PerformedPart)):
    elif isinstance(fn_spart_or_ppart, (Score, PerformedPart)):
        part = fn_spart_or_ppart
    else:
        part = fn_spart_or_ppart
  
    s_note_array = part.note_array()
    if isinstance(part, Part):
        p_note_array = performance_from_part(part, bpm).note_array()
        time_signature_map = part.time_signature_map
    else:
        p_note_array = s_note_array
        time_signature_map = None

    notes = []
    for sn, pn in zip(s_note_array, p_note_array):
        note = Note(
            pitch=sn["pitch"],
            onset=sn["onset_beat"],
            duration=sn["duration_beat"],
            p_onset=pn["onset_sec"],
            p_duration=pn["duration_sec"],
            velocity=pn["velocity"],
            id=sn["id"],
        )
        notes.append(note)

    score = Score(
        notes,
        time_signature_map=time_signature_map,
        note_array=s_note_array,
    )

    return score


def alignment_to_score(fn_or_spart, ppart, alignment):
    """
    Get a `Score` instance from a partitura `Part` or `PerformedPart`

    Parameters
    ----------
    fn_spart_or_ppart : filename, Part of PerformedPart
        Filename or partitura Part
    bpm : float
        Beats per minute to generate the performance (this is ignored if
        the input is a `PerformedPart`
    velocity : int
        Velocity to generate the performance (this is ignored if the input
        is a `PerformedPart`

    Returns
    -------
    score : Score
        A `Score` object.
    """

    if isinstance(fn_or_spart, str):
        part = load_score(fn_or_spart)
    elif isinstance(fn_or_spart, Part):
        part = fn_or_spart
    else:
        raise ValueError(
            "`fn_or_spart` must be a `Part` or a filename, " f"but is {type(part)}."
        )

    if not isinstance(ppart, PerformedPart):
        raise ValueError(
            "`ppart` must be a `PerformedPart`, but is ", f"{type(ppart)}."
        )

    part_by_id = dict((n.id, n) for n in part.notes_tied)

    ppart_by_id = dict((n["id"], n) for n in ppart.notes)

    note_pairs = [
        (part_by_id[a["score_id"]], ppart_by_id[a["performance_id"]])
        for a in alignment
        if (a["label"] == "match" and a["score_id"] in part_by_id)
    ]

    notes = []
    pitch_onset = [(sn.midi_pitch, sn.start.t) for sn, _ in note_pairs]
    sort_order = np.lexsort(list(zip(*pitch_onset)))
    beat_map = part.beat_map
    for i in sort_order:
        sn, n = note_pairs[i]
        sn_on, sn_off = beat_map([sn.start.t, sn.start.t + sn.duration_tied])
        sn_dur = sn_off - sn_on
        # hack for notes with negative durations
        n_dur = max(n["sound_off"] - n["note_on"], 60 / 200 * 0.25)
        note = Note(
            pitch=sn.midi_pitch,
            onset=sn_on,
            duration=sn_dur,
            p_onset=n["note_on"],
            p_duration=n_dur,
            id=sn.id,
        )
        notes.append(note)

    score = Score(notes, time_signature_map=part.time_signature_map)

    return score


if __name__ == "__main__":

    import partitura

    fn = "../demo_data/twinkle_twinkle_little_star_score.musicxml"

    spart = partitura.load_musicxml(fn)

    score = Score(spart)
