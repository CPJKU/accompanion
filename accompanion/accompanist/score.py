# -*- coding: utf-8 -*-
"""
Objects for representing score information
"""
import numpy as np
import mido

from mido import Message


class Note(object):
    """
    Class for representing notes
    """

    def __init__(self, pitch, onset, duration, p_onset=None,
                 p_duration=None, velocity=64):

        self.pitch = pitch
        self.onset = onset
        self.duration = duration
        self.p_onset = p_onset
        self.p_duration = p_duration
        self.already_performed = False
        self.velocity = velocity
        self._note_on = Message(
            type='note_on',
            velocity=self.velocity,
            note=self.pitch,
            time=self.p_onset if self.p_onset is not None else 0)

        self._note_off = Message(
            type='note_off',
            velocity=self.velocity,
            note=self.pitch,
            time=self.p_offset)

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
        elif len(p_onset) == self.num_notes:
            for n, po in zip(self.notes, p_onset):
                n.p_onset = po

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
        elif len(p_duration) == self.num_notes:
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
        elif len(velocity) == self.num_notes:
            for n, po in zip(self.notes, velocity):
                n.velocity = po


class ExpressiveChord(Chord):
    def __init__(self, notes,
                 velocity_trend=None,
                 velocity_dev=None,
                 log_bpr=None,
                 timing=None,
                 log_articulation=None):
        super().__init__(notes)

        self.velocity_trend = velocity_trend
        self.velocity_dev = velocity_dev
        self.log_bpr = log_bpr
        self.timing = timing
        self.log_articulation = log_articulation


class Score(object):

    def __init__(self, notes, access_mode='indexwise'):

        # TODO: Seconday sort by pitch
        self.notes = np.array(sorted(notes, key=lambda x: x.pitch))

        self.access_mode = access_mode
        onsets = np.array([n.onset for n in self.notes])

        # Unique score positions
        self.unique_onsets = np.unique(onsets)
        self.unique_onsets.sort()

        # indices of the notes belonging to each
        self.unique_onset_idxs = [np.where(onsets == u)
                                  for u in self.unique_onsets]

        self.chords = np.array(
            [Chord(self.notes[ui])
             for ui in self.unique_onset_idxs])

        self.chord_dict = dict(
            [(u, c) for u, c in zip(self.unique_onsets, self.chords)])

    @property
    def access_mode(self):
        return self._access_mode

    @access_mode.setter
    def access_mode(self, access_mode):
        if access_mode not in ('indexwise', 'timewise'):
            raise ValueError('`access_mode` should be "indexwise" or "timewise". Given {0}'.format(access_mode))
        self._access_mode = access_mode

    def __getitem__(self, index):
        if self.access_mode == 'indexwise':
            return self.chords[index]
        elif self.access_mode == 'timewise':
            return self.chord_dict[index]

    def __len__(self):
        return len(self.unique_onsets)


class AccompanimentScore(Score):

    def __init__(self, notes, solo_score,
                 mode='iter_solo',
                 velocity_trend=None,
                 velocity_dev=None,
                 log_bpr=None,
                 timing=None,
                 log_articulation=None):
        super().__init__(notes, access_mode='indexwise')

        assert isinstance(solo_score, Score)

        self.ssc = solo_score

        self.mode = mode

        self.solo_score_dict = dict()

        self.next_onsets = dict()

        self.velocity_trend = dict()
        self.velocity_dev = dict()
        self.log_bpr = dict()
        self.timing = dict()
        self.log_articulation = dict()

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

                self.next_onsets[on] = (self.chords[acc_idx:], next_iois)

                self.solo_score_dict[on] = (self.chords[acc_idx:],
                                            next_iois,
                                            next_acc_onsets,
                                            acc_idx,
                                            i)
            else:
                self.next_onsets[on] = (None, None)
                self.solo_score_dict = (None,
                                        None,
                                        None,
                                        None,
                                        i)


if __name__ == '__main__':

    import partitura

    solo_fn = '/Users/aae/Repos/accompanion_poly/data/Ravel_Menuet/Minuet-Ravel-4hands-Primo.musicxml'

    acc_fn = '/Users/aae/Repos/accompanion_poly/data/Ravel_Menuet/Minuet-Ravel-4hands-Secondo.musicxml'

    acc_array = partitura.musicxml.xml_to_notearray(acc_fn)
    solo_array = partitura.musicxml.xml_to_notearray(solo_fn)

    p_onsets = acc_array['onset'] + 0.05 * np.random.randn(len(acc_array))
    p_onsets -= p_onsets.min()

    acc_notes = [Note(n['pitch'] - 12, n['onset'], n['duration'],
                      po,
                      n['duration'], 30)
                 for n, po in zip(acc_array, p_onsets)]

    solo_notes = [Note(n['pitch'] - 12, n['onset'], n['duration'],
                       n['onset'],
                       n['duration'], 30)
                  for n in solo_array]

    solo_score = Score(solo_notes)
    acc_score = AccompanimentScore(acc_notes,
                                   solo_score)
