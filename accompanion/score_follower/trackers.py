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
        # print(score_positions)
        score_position = np.median(score_positions)
        self.current_position = score_position

        if self.update_sf_positions:
            self.update_position(score_position)

        # todo @carlos which index needs to be chosen?
        index = indices[0]

        return index, score_position

    def update_position(self, ref_time):
        for sf, rtsm in zip(self.score_followers, self.ref_to_state_time_maps):
            st = rtsm(ref_time) * self.inv_polling_period
            # print('sf async', st - sf.current_position)
            sf.current_position = int(np.round(st))
            # sf.restart = True


class GroundTruthTracker(object):
    """
    Does this tracker still works?

    TODO
    ----
    * check if this class works, and if not either adapt it or delete it
    """
    def __init__(self, match_fn, frame_resolution, score_bpm, pipeline, **kwargs):
        ppart, gt_alignment, spart = partitura.load_match(
            match_fn, create_part=True, first_note_at_zero=True
        )

        _, ref_onsets, _, _, reference = partitura_to_framed_midi_custom(
            part_or_notearray_or_filename=spart,
            polling_period=frame_resolution,
            pipeline=pipeline,
            score_bpm=score_bpm,
            tempo_curve=None,
            return_reference=True,
        )

        _, perf_onsets, _, _ = partitura_to_framed_midi_custom(
            part_or_notearray_or_filename=ppart,
            polling_period=frame_resolution,
            pipeline=pipeline,
            is_performance=True,
            return_reference=False,
        )

        # Get matched notes
        matched_idxs = get_matched_notes(
            reference.note_array, ppart.note_array, gt_alignment
        )

        # Get onset frames of only matched notes
        ref_onsets = ref_onsets[matched_idxs[:, 0]]
        perf_onsets = perf_onsets[matched_idxs[:, 1]]

        ref_onset_frames = (ref_onsets / frame_resolution).astype(np.int)
        perf_onset_frames = (perf_onsets / frame_resolution).astype(np.int)

        unique_perf_onset_frames = np.unique(perf_onset_frames)
        perf_onset_idxs = [
            np.where(perf_onset_frames == u)[0] for u in unique_perf_onset_frames
        ]

        perf_onset_frames = unique_perf_onset_frames
        ref_onset_frames = np.array(
            [ref_onset_frames[uix].min() for uix in perf_onset_idxs]
        )

        assert len(perf_onset_frames) == len(ref_onset_frames)
        self.input_index = 0
        self.current_position = 0
        self.gt = interpolate.interp1d(
            perf_onset_frames,
            ref_onset_frames,
            kind="linear",
            bounds_error=False,
            fill_value=(ref_onset_frames[0], ref_onset_frames[-1]),
        )

    def __call__(self, input):
        self.step(input)
        return self.current_position

    def step(self, input_features):
        """
        Step
        """
        self.current_position = int(self.gt(self.input_index))
        self.input_index += 1
