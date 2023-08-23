# -*- coding: utf-8 -*-
"""
TODO
----
* Update the MultiDTWScoreFollower for HMM?
"""
import numpy as np


class AccompanimentScoreFollower(object):
    """
    Parent Class for all Accompaniment Score Followers.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, frame):
        raise NotImplementedError

    def update_position(self, ref_time):
        pass


class HMMScoreFollower(AccompanimentScoreFollower):
    """
    An HiddenMarkov Model based Score Follower.

    Parameters
    ----------
    score : partitura.score.Part
        The score to be followed.
    """

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


class MultiDTWScoreFollower(AccompanimentScoreFollower):
    """
    A Multi Dynamic Time Warping based Score Follower.

    Parameters
    ----------
    score_followers: list
        A list of Score Followers to be used.
    state_to_ref_time_maps: list
        A list of State to Reference Time Maps to be used.
    ref_to_state_time_maps: list
        A list of Reference Time to State Maps to be used.
    polling_period: float
        The polling period of the Score Followers.
        Polling period (in seconds) used to convert the MIDI messages
    update_sf_positions: bool
        Whether to update the Score Follower positions or not.
    """

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
        """
        Get score position by aggregating the predicted position of all
        followers in the ensemble
        """
        score_positions = []
        for sf, strm in zip(self.score_followers, self.state_to_ref_time_maps):
            st = sf(frame)
            sp = float(strm(st * self.polling_period))
            score_positions.append(sp)
        score_position = np.median(score_positions)
        self.current_position = score_position

        if self.update_sf_positions:
            # Update the position in the individual score followers
            self.update_position(score_position)

        return score_position

    def update_position(self, ref_time):
        """
        Update the current position in each of the score followers

        Parameters
        ----------
        ref_time : float
            Current time in the score follower
        """
        for sf, rtsm in zip(self.score_followers, self.ref_to_state_time_maps):
            st = rtsm(ref_time) * self.inv_polling_period
            sf.current_position = int(np.round(st))
