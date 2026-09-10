# -*- coding: utf-8 -*-
"""
Adapters between Matchmaker's score followers and the ACCompanion.

The alignment algorithms themselves live in the `matchmaker` package
(https://github.com/pymatchmaker/matchmaker). The classes in this module only
translate between Matchmaker's interface, where a follower is called as
``follower(features, perf_time)`` and returns a score position in beats, and the
interface expected by the ACCompanion main loop, where a follower is called with
the output of the input pipeline and returns a score position, or None whenever
it has no new position to report.
"""
import time
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from matchmaker.dp.oltw_arzt import OnlineTimeWarpingArztFrame
from matchmaker.prob.hmm import BaseHMM

from accompanion.accompanist.tempo_models import SyncModel

# Output of a Matchmaker feature processor: a (features, performance time)
# tuple, or None while the processor has nothing to report (e.g., a frame
# without note onsets for the pitch processor).
ProcessorOutput = Optional[Tuple[np.ndarray, float]]

TimeMap = Callable[[Union[float, int, np.ndarray]], Union[float, int, np.ndarray]]


class AccompanimentScoreFollower(object):
    """
    Parent Class for all Accompaniment Score Followers.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, observation: ProcessorOutput) -> Optional[float]:
        raise NotImplementedError

    def update_position(self, ref_time: float) -> None:
        pass

    def discount_wait(self, perf_time: float, expected_ioi: float) -> None:
        """Show a wait to the follower as the interval the score expects.

        Called on every frame the accompaniment spends waiting at a fermata.
        The gap a soloist opens there is not an inter-onset interval any tempo
        can explain, and a follower whose observation model reads it as one
        will place them wherever that much elapsed time would have taken them
        -- which, after a three-second fermata, is most of a bar further on
        than they actually are.

        What the follower should see instead is the interval the score writes
        between the fermata and the note after it: whenever the soloist gets
        round to playing that note, the step they are making is the notated
        one.

        Parameters
        ----------
        perf_time : float
            The time of this frame, in seconds.
        expected_ioi : float
            The notated interval to the onset that will end the wait, in
            seconds at the current tempo.

        Notes
        -----
        Does nothing by default: a follower that does not model timing has
        nothing to correct, and one that measures its own inter-onset
        intervals internally cannot be corrected from here.
        """
        pass


class HMMScoreFollower(AccompanimentScoreFollower):
    """
    A Hidden Markov Model based Score Follower.

    Wraps a Matchmaker HMM (`matchmaker.prob.hmm.BaseHMM`) whose observation
    model expects a ``(pitch_obs, ioi_obs, beat_period)`` observation, i.e.
    Matchmaker's `ACCPitchIOIObservationModel`. The inter-onset interval is
    derived here from the performance times reported by the input processor.

    Parameters
    ----------
    score_follower: BaseHMM
        The Matchmaker HMM to be used.
    update_tempo_model: bool
        If True, the tempo model of the score follower is updated on every
        score onset (the behavior of the former `PitchIOIKHMM`). If False, the
        tempo model is only read, and is expected to be updated elsewhere
        (the behavior of the former `PitchIOIHMM`, whose tempo model is the
        one shared with the accompanist).
    """

    def __init__(
        self,
        score_follower: BaseHMM,
        update_tempo_model: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.score_follower: BaseHMM = score_follower
        self.update_tempo_model: bool = update_tempo_model
        self.current_position: float = 0
        self.prev_perf_time: Optional[float] = None

    def discount_wait(self, perf_time: float, expected_ioi: float) -> None:
        """Cap the interval the follower will be shown at the notated one.

        `prev_perf_time` is only ever moved forward, so a wait longer than the
        score asks for is trimmed back to it and a wait shorter than it is
        left alone. The follower is never told that *more* time has passed
        than really has -- which, at a fermata the soloist declines to hold,
        would push it forward for nothing.
        """
        if self.prev_perf_time is not None:
            self.prev_perf_time = max(self.prev_perf_time, perf_time - expected_ioi)

    def __call__(self, observation: ProcessorOutput) -> Optional[float]:
        if observation is None:
            return None

        pitch_obs, perf_time = observation

        # Inter-onset interval since the previous observed onset
        ioi_obs = (
            perf_time - self.prev_perf_time if self.prev_perf_time is not None else 0.0
        )
        self.prev_perf_time = perf_time

        tempo_model = self.score_follower.tempo_model
        score_position = self.score_follower(
            (pitch_obs, ioi_obs, tempo_model.beat_period),
            perf_time,
        )

        # With inserted states, the odd-numbered states stand for insertions
        # (i.e., notes that are not in the score), and only the even-numbered
        # ones correspond to an actual score onset.
        if (
            self.score_follower.has_insertions
            and self.score_follower.current_index % 2 == 0
        ):
            if self.update_tempo_model:
                tempo_model.update_beat_period(
                    performed_onset=perf_time,
                    score_onset=score_position,
                )
                tempo_model.counter += 1

            self.current_position = score_position
            return self.current_position

        return None


class MatchmakerScoreFollower(AccompanimentScoreFollower):
    """
    Adapter for any of Matchmaker's score followers.

    Every Matchmaker follower implements the same contract -- called with
    ``(features, perf_time)``, it returns the current score position in beats --
    so this one class drives all of them, including followers added to
    Matchmaker after this code was written.

    Parameters
    ----------
    score_follower: matchmaker.base.OnlineAlignment
        The follower to be driven, as built by
        `accompanion.score_follower.matchmaker_methods.build_score_follower`.
    raw_modality: str, optional
        Set for a composite follower, such as Matchmaker's ensemble, which is
        handed the untouched stream frame tagged with its modality rather than
        the features of a single processor. It applies its members' own
        processors itself.
    """

    def __init__(self, score_follower, raw_modality=None, **kwargs) -> None:
        super().__init__()
        self.score_follower = score_follower
        self.raw_modality: Optional[str] = raw_modality
        self.current_position: float = float(
            getattr(score_follower, "current_position", 0.0)
        )

    def __call__(self, observation: ProcessorOutput) -> Optional[float]:
        if observation is None:
            # The processor is still buffering, or the frame held no onset.
            return None

        features, perf_time = observation
        if self.raw_modality is not None:
            features = (self.raw_modality, features)
        position = float(self.score_follower(features, perf_time))

        # HMM followers interleave an "insertion" state between consecutive
        # score onsets, to absorb notes that are not in the score. A position
        # reported on one of those is not a score onset, so the ACCompanion
        # must not act on it.
        if (
            getattr(self.score_follower, "has_insertions", False)
            and self.score_follower.current_index % 2 != 0
        ):
            return None

        self.current_position = position
        return position

    def update_position(self, ref_time: float) -> None:
        """
        Move the follower to `ref_time` after the soloist has gone silent.

        Matchmaker followers re-anchor themselves through `set_position`, which
        each one implements in terms of its own state -- an HMM re-concentrates
        its belief, an on-line time warper reseeds its cost matrix. Older
        Matchmaker versions have no such method, in which case the follower is
        moved to the state nearest `ref_time` and left to recover on its own.
        """
        set_position = getattr(self.score_follower, "set_position", None)
        if callable(set_position):
            set_position(ref_time)
            return

        positions = getattr(self.score_follower, "score_positions", None)
        if positions is None or len(positions) == 0:
            return

        index = int(np.searchsorted(positions, ref_time))
        self.score_follower.current_index = int(
            np.clip(index, 0, len(positions) - 1)
        )


class MultiDTWScoreFollower(AccompanimentScoreFollower):
    """
    A Multi Dynamic Time Warping based Score Follower.

    The individual followers are Matchmaker on-line time warping followers,
    each of which aligns the input against one reference performance and
    reports its position directly in score beats.

    Parameters
    ----------
    score_followers: list
        A list of Matchmaker Score Followers to be used.
    ref_to_state_time_maps: list
        A list of maps from reference time (score beats) to performance time
        (seconds) for each of the score followers. They are used to reset the
        position of the followers in `update_position`.
    polling_period: float
        Polling period (in seconds) used to convert the MIDI messages, i.e. the
        duration of a single frame of the reference features.
    update_sf_positions: bool
        Whether to update the Score Follower positions or not.
    """

    def __init__(
        self,
        score_followers: List[OnlineTimeWarpingArztFrame],
        ref_to_state_time_maps: List[TimeMap],
        polling_period: float,
        update_sf_positions: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.score_followers: List[OnlineTimeWarpingArztFrame] = score_followers
        self.ref_to_state_time_maps: List[TimeMap] = ref_to_state_time_maps
        self.polling_period: float = polling_period
        self.inv_polling_period: float = 1 / polling_period
        self.update_sf_positions: bool = update_sf_positions
        self.current_position: float = 0

    def __call__(self, observation: ProcessorOutput) -> float:
        """
        Get score position by aggregating the predicted position of all
        followers in the ensemble
        """
        features, perf_time = observation

        score_positions = [sf(features, perf_time) for sf in self.score_followers]
        score_position = float(np.median(score_positions))
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
            set_position = getattr(sf, "set_position", None)
            if callable(set_position):
                # Maps the beat back to a reference frame and reseeds the cost
                # matrix, so the search window recenters on the next step.
                set_position(ref_time)
                continue

            frame_index = int(np.round(float(rtsm(ref_time)) * self.inv_polling_period))
            # `_current_frame` is the index into the reference features around
            # which the follower centers its search window.
            sf._current_frame = int(np.clip(frame_index, 0, sf.N_ref - 1))


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
