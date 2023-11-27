# -*- coding: utf-8 -*-
"""
TODO
----
* Update the MultiDTWScoreFollower for HMM?
"""
import time
from typing import Callable, List, Optional, Union

import numpy as np

from accompanion.accompanist.tempo_models import SyncModel
from accompanion.mtchmkr.alignment_online_oltw import OnlineTimeWarping
from accompanion.mtchmkr.score_hmm import PitchIOIHMM, PitchIOIKHMM

HMM_SF_Types = Union[PitchIOIKHMM, PitchIOIHMM]


class AccompanimentScoreFollower(object):
    """
    Parent Class for all Accompaniment Score Followers.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, frame: Optional[np.ndarray]) -> Optional[float]:
        raise NotImplementedError

    def update_position(self, ref_time: float) -> None:
        pass


class HMMScoreFollower(AccompanimentScoreFollower):
    """
    An HiddenMarkov Model based Score Follower.

    Parameters
    ----------
    score_followers: Union[PitchIOIKHMM, PitchIOIKHMM]
        The score follower to be used.
    """

    def __init__(self, score_follower: HMM_SF_Types, **kwargs) -> None:
        super().__init__()
        self.score_follower: HMM_SF_Types = score_follower
        self.current_position: int = 0

    def __call__(self, frame: Optional[np.ndarray]) -> Optional[float]:
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
        score_followers: List[OnlineTimeWarping],
        state_to_ref_time_maps: List[
            Callable[[Union[float, int, np.ndarray]], Union[float, int, np.ndarray]]
        ],
        ref_to_state_time_maps: List[
            Callable[[Union[float, int, np.ndarray]], Union[float, int, np.ndarray]]
        ],
        polling_period: float,
        update_sf_positions: bool = False,
        *kwargs,
    ) -> None:
        super().__init__()
        self.score_followers: List[OnlineTimeWarping] = score_followers
        self.state_to_ref_time_maps: List[
            Callable[[Union[float, int, np.ndarray]], Union[float, int, np.ndarray]]
        ] = state_to_ref_time_maps
        self.ref_to_state_time_maps: List[
            Callable[[Union[float, int, np.ndarray]], Union[float, int, np.ndarray]]
        ] = ref_to_state_time_maps
        self.polling_period: float = polling_period
        self.inv_polling_period: float = 1 / polling_period
        self.update_sf_positions: bool = update_sf_positions
        self.current_position: int = 0

    def __call__(self, frame: Optional[np.ndarray]) -> float:
        """
        Get score position by aggregating the predicted position of all
        followers in the ensemble
        """
        score_positions = []
        predicted_frames = []
        for sf, strm in zip(self.score_followers, self.state_to_ref_time_maps):
            st = sf(frame)
            sp = float(strm(st * self.polling_period))
            score_positions.append(sp)
            predicted_frames.append(st)
        score_position = np.median(score_positions)
        # print("predicted_frames", predicted_frames)
        # print("score_positions", score_positions)
        self.current_position = score_position

        if self.update_sf_positions:
            # Update the position in the individual score followers
            self.update_position(score_position)

        return score_position

    def update_position(self, ref_time: float) -> None:
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


class ExpectedPositionTracker(object):
    tempo_model: SyncModel
    prev_position: float = None
    prev_time: Optional[float] = None

    def __init__(self, tempo_model: SyncModel, first_onset: float) -> None:
        self.tempo_model = tempo_model
        self.prev_position = first_onset

    @property
    def expected_position(self) -> float:
        current_time = time.time()

        if self.prev_time is None:
            self.prev_time = current_time
            return self.prev_position
        else:
            # inter-event-interval
            iei = current_time - self.prev_time
            expected_position = self.prev_position + iei / max(
                self.tempo_model.beat_period, 1e-6
            )
            self.prev_position = expected_position
            return expected_position

    @expected_position.setter
    def expected_position(self, score_position: float) -> None:
        self.prev_position = score_position
        self.prev_time = time.time()
