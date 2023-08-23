from typing import Tuple, List, Optional
import numpy as np


class OnsetTracker(object):
    """
    A class to keep track of the performed onsets and knows at all times
    what is the current score onset and what is the next onset.

    Parameters
    ----------
    unique_onsets : np.ndarray
        The unique score onsets in beats.
    min_acc_delta : float
    """

    def __init__(self, unique_onsets: np.ndarray, min_acc_delta: float = 2) -> None:
        self.unique_onsets: np.ndarray = unique_onsets
        self.max_idx: int = len(self.unique_onsets) - 1
        self.unique_onsets.sort()
        self.current_idx: int = 0
        self.performed_onsets: List[float] = []
        self.min_acc_delta: float = min_acc_delta

    def __call__(
        self,
        score_time: Optional[float],
        acc_score_time: float = -np.inf,
    ) -> Tuple[Optional[float], Optional[int], bool]:
        solo_s_onset: Optional[float] = None
        onset_index: Optional[int] = None
        acc_update: Optional[bool] = False

        if score_time is not None:
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

    def is_acc_update(self, acc_score_time: float) -> bool:
        next_onset_crit: bool = acc_score_time >= self.next_onset
        delta_onset_crit: bool = (
            acc_score_time - self.current_onset >= self.min_acc_delta
        )
        return next_onset_crit and delta_onset_crit

    @property
    def current_onset(self) -> float:
        try:
            return self.unique_onsets[self.current_idx]
        except IndexError:
            return self.unique_onsets[-1]

    def acc_onset(self, acc_score_time) -> Tuple[float, int]:
        min_idx: int = np.argmax(
            acc_score_time >= self.unique_onsets[self.current_idx :]
        )  # np.where(acc_score_time >= self.unique_onsets)[0]
        try:
            c_idx: int = self.current_idx + min_idx
            return self.unique_onsets[c_idx], c_idx
        except (ValueError, IndexError):
            return self.unique_onsets[-1], self.max_idx

    @property
    def next_onset(self) -> float:
        try:
            return self.unique_onsets[self.current_idx + 1]
        except IndexError:
            return self.unique_onsets[-1]


class DiscreteOnsetTracker(object):
    """
    A class to track discrete onset events.

    Parameters
    ----------
    unique_onsets : np.ndarray
        The unique score onsets in beats.
    """
    def __init__(self, unique_onsets: np.ndarray, *args, **kwargs) -> None:
        print("Using discrete onset tracker")
        self.unique_onsets = unique_onsets
        self.max_idx: int = len(self.unique_onsets) - 1
        self.unique_onsets.sort()
        self.current_idx: int = 0
        self.performed_onsets: List[float] = []

        self.idx_dict = dict([(uo, i) for i, uo in enumerate(self.unique_onsets)])
        self.current_onset = self.unique_onsets[0]

    def __call__(
        self,
        score_time: Optional[float],
        acc_score_time: float = -np.inf,
    ) -> Tuple[Optional[float], Optional[int], bool]:
        solo_s_onset: Optional[float] = None
        onset_index: Optional[int] = None
        acc_update: Optional[bool] = False

        if score_time is not None:
            if (
                score_time in self.unique_onsets
                and score_time not in self.performed_onsets
            ):
                solo_s_onset = score_time
                self.current_idx = self.idx_dict[solo_s_onset]
                self.current_onset = solo_s_onset
                self.performed_onsets.append(score_time)
                onset_index = self.current_idx

                print(f'onset tracker {score_time}')

        return solo_s_onset, onset_index, acc_update
