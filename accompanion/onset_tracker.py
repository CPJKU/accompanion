import numpy as np

class OnsetTracker(object):
    def __init__(self, unique_onsets):
        self.unique_onsets = unique_onsets
        self.max_idx = len(self.unique_onsets) - 1
        self.unique_onsets.sort()
        self.current_idx = 0
        self.performed_onsets = []
        self.min_acc_delta = 2

    def __call__(self, score_time, acc_score_time=-np.inf):
        solo_s_onset = None
        onset_index = None
        acc_update = False

        if score_time >= self.current_onset:
            if self.current_onset not in self.performed_onsets:
                solo_s_onset = self.current_onset
                onset_index = self.current_idx
                self.performed_onsets.append(self.current_onset)
                self.current_idx += 1

        if self.is_acc_update(acc_score_time):
            acc_onset, cidx = self.acc_onset(acc_score_time)
            if acc_onset not in self.performed_onsets:
                solo_s_onset, onset_index = acc_onset, cidx
                self.performed_onsets.append(acc_onset)
                self.current_idx = onset_index + 1
                acc_update = True

        return solo_s_onset, onset_index, acc_update

    def is_acc_update(self, acc_score_time):
        next_onset_crit = acc_score_time >= self.next_onset
        delta_onset_crit = acc_score_time - self.current_onset >= self.min_acc_delta
        return next_onset_crit and delta_onset_crit

    @property
    def current_onset(self):
        try:
            return self.unique_onsets[self.current_idx]
        except IndexError:
            return self.unique_onsets[-1]

    def acc_onset(self, acc_score_time):
        min_idx = np.argmax(acc_score_time >= self.unique_onsets[self.current_idx:])# np.where(acc_score_time >= self.unique_onsets)[0]
        try:
            c_idx = self.current_idx + min_idx
            return self.unique_onsets[c_idx], c_idx
        except (ValueError, IndexError):
            return self.unique_onsets[-1], self.max_idx

    @property
    def next_onset(self):
        try:
            return self.unique_onsets[self.current_idx + 1]
        except IndexError:
            return self.unique_onsets[-1]
