"""
TODO
----
* Update the MultiDTWScoreFollower for HMM?
"""
import partitura

import numpy as np

from accompanion.utils.partitura_utils import (
    get_matched_notes,
    partitura_to_framed_midi_custom,
)
from scipy import interpolate


class AccompanimentScoreFollower(object):
    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        raise NotImplementedError


class HMMScoreFollower(AccompanimentScoreFollower):
    def __init__(self, score_follower, update_sf_positions=False):
        super().__init__()
        self.score_follower = score_follower
        self.current_position = 0

    def __call__(self, frame):

        if frame is not None:
            current_position = self.score_follower(frame)

            if (
                self.score_follower.has_insertions
                and self.score_follower.current_state % 2 == 0
            ):
                self.current_position = current_position
                return self.current_position
        return None
        # return self.score_follower.current_state, self.score_position

    def update_position(self, ref_time):
        pass


class MultiDTWScoreFollower(AccompanimentScoreFollower):
    def __init__(
        self,
        score_followers,
        state_to_ref_time_maps,
        ref_to_state_time_maps,
        polling_period,
        update_sf_positions=False,
    ):
        super().__init__()
        self.score_followers = score_followers
        self.state_to_ref_time_maps = state_to_ref_time_maps
        self.ref_to_state_time_maps = ref_to_state_time_maps
        self.polling_period = polling_period
        self.inv_polling_period = 1 / polling_period
        self.update_sf_positions = update_sf_positions
        self.current_position = 0

    def __call__(self, frame):

        score_positions = []
        indices = []
        for sf, strm in zip(self.score_followers, self.state_to_ref_time_maps):
            st = sf(frame)
            indices.append(st)
            sp = float(strm(st * self.polling_period))
            score_positions.append(sp)
        score_position = np.median(score_positions)
        self.current_position = score_position

        if self.update_sf_positions:
            self.update_position(score_position)

        return score_position

    def update_position(self, ref_time):
        for sf, rtsm in zip(self.score_followers, self.ref_to_state_time_maps):
            st = rtsm(ref_time) * self.inv_polling_period
            sf.current_position = int(np.round(st))

