# -*- coding: utf-8 -*-
"""
Online Time Warping ACCompanion.

This module contains the main class for the Online Time Warping ACCompanion.
It works as a follower for complicated pieces usually for four hands.
"""
import os
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import partitura as pt
from basismixer.performance_codec import get_performance_codec
from partitura.musicanalysis.performance_codec import (
    get_matched_notes,
    get_time_maps_from_alignment,
    notewise_to_onsetwise,
    onsetwise_to_notewise,
)
from partitura.performance import PerformedPart

# from basismixer.utils.music import onsetwise_to_notewise, notewise_to_onsetwise
from scipy.interpolate import interp1d

from accompanion.accompanist import tempo_models
from accompanion.accompanist.accompaniment_decoder import moving_average_offline
from accompanion.accompanist.score import (
    AccompanimentScore,
    alignment_to_score,
    part_to_score,
)
from accompanion.base import ACCompanion
from accompanion.midi_handler.midi_input import POLLING_PERIOD
from accompanion.mtchmkr.alignment_online_oltw import OnlineTimeWarping
from accompanion.mtchmkr.features_midi import PianoRollProcessor
from accompanion.mtchmkr.utils_generic import SequentialOutputProcessor
from accompanion.score_follower.trackers import MultiDTWScoreFollower
from accompanion.utils.partitura_utils import (
    partitura_to_framed_midi_custom as partitura_to_framed_midi,
)
from accompanion.utils.partitura_utils import performance_notearray_from_score_notearray

SCORE_FOLLOWER_DEFAULT_KWARGS = {
    "score_follower": "OnlineTimeWarping",
    "window_size": 80,
    "step_size": 10,
    "input_processor": {
        "processor": "PianoRollProcessor",
        "processor_kwargs": {"piano_range": True},
    },
}

TEMPO_MODEL_DEFAULT_KWARGS = {"tempo_model": tempo_models.LSM}

PERFORMANCE_CODEC_DEFAULT_KWARGS = {
    "velocity_trend_ma_alpha": 0.6,
    "articulation_ma_alpha": 0.4,
    "velocity_dev_scale": 70,
    "velocity_min": 20,
    "velocity_max": 100,
    "velocity_solo_scale": 0.85,
    "timing_scale": 0.001,
    "log_articulation_scale": 0.1,
    "mechanical_delay": 0.0,
}


class OLTWACCompanion(ACCompanion):
    """
    The On-Line Time Wrapping Accompanion Follower Class.
    It inherits from the Base ACCompanion Class. It updates the methods
    setup_scores, setup_score_follower, and check_empty_frames.

    Parameters
    ----------
    solo_fn : str
        The path to the solo score.
    acc_fn : str
        The path to the accompaniment score.
    midi_router_kwargs : dict
        The keyword arguments for the MIDI Router.
    accompaniment_match : str, optional
        The path to the accompaniment match file, by default None.
    midi_fn : str, optional
        The path to the MIDI file, by default None.
    score_follower_kwargs : dict, optional
        The keyword arguments for the score follower, by default {"score_follower": "PitchIOIHMM", "score_follower_kwargs": {}, "input_processor": {"processor": "PitchIOIProcessor", "processor_kwargs": {}}}.
    tempo_model_kwargs : dict, optional
        The keyword arguments for the tempo model, by default {"tempo_model": tempo_models.LTSM}.
    performance_codec_kwargs : dict, optional
        The keyword arguments for the performance codec, by default {"velocity_trend_ma_alpha": 0.6, "articulation_ma_alpha": 0.4, "velocity_dev_scale": 70, "velocity_min": 20, "velocity_max": 100, "velocity_solo_scale": 0.85, "timing_scale": 0.001, "log_articulation_scale": 0.1, "mechanical_delay": 0.0}.
    init_bpm : float, optional
        The initial BPM, by default 60.
    init_velocity : int, optional
        The initial velocity, by default 60.
    polling_period : float, optional
        The polling period, by default POLLING_PERIOD.
    use_ceus_mediator : bool, optional
        Whether to use the CEUS Mediator, by default False.
    adjust_following_rate : float, optional
        The adjustment rate for the following rate, by default 0.1.
    bypass_audio : bool, optional
        Whether to bypass the audio, by default False.
    test : bool, optional
        Whether to bypass the MIDI Router, by default False.
    record_midi : bool, optional
        Whether to record the MIDI, by default False.
    """

    def __init__(
        self,
        solo_fn: str,
        acc_fn: str,
        midi_router_kwargs: Dict[str, Any],  # this is just a workaround for now
        accompaniment_match: Optional[str] = None,
        midi_fn: Optional[str] = None,
        score_follower_kwargs: Dict[
            str, Union[str, float, int, dict]
        ] = SCORE_FOLLOWER_DEFAULT_KWARGS,
        tempo_model_kwargs: Dict[
            str, Union[str, float, int, dict, tempo_models.SyncModel]
        ] = TEMPO_MODEL_DEFAULT_KWARGS,
        performance_codec_kwargs: Dict[
            str, Union[float, int, str]
        ] = PERFORMANCE_CODEC_DEFAULT_KWARGS,
        init_bpm: float = 60,
        init_velocity: int = 60,
        polling_period: float = POLLING_PERIOD,
        use_ceus_mediator: bool = False,
        adjust_following_rate: float = 0.1,
        expected_position_weight: float = 0.6,
        bypass_audio: bool = False,  # bypass fluidsynth audio
        test: bool = False,  # bypass MIDIRouter
        record_midi: bool = False,
        accompanist_decoder_kwargs: Optional[
            Dict[str, Union[float, int, str, dict]]
        ] = None,
    ) -> None:
        # Remember that strings are also iterables ;)
        score_kwargs = dict(
            solo_fn=solo_fn
            if (isinstance(solo_fn, Iterable) and not isinstance(solo_fn, str))
            else [solo_fn],
            acc_fn=acc_fn,
            accompaniment_match=accompaniment_match,
        )
        super().__init__(
            score_kwargs=score_kwargs,
            score_follower_kwargs=score_follower_kwargs,
            performance_codec_kwargs=performance_codec_kwargs,
            midi_router_kwargs=midi_router_kwargs,
            midi_fn=midi_fn,
            init_bpm=init_bpm,
            init_velocity=init_velocity,
            polling_period=polling_period,
            use_ceus_mediator=use_ceus_mediator,
            adjust_following_rate=adjust_following_rate,
            expected_position_weight=expected_position_weight,
            bypass_audio=bypass_audio,
            tempo_model_kwargs=tempo_model_kwargs,
            test=test,
            record_midi=record_midi,
            accompanist_decoder_kwargs=accompanist_decoder_kwargs,
        )

        self.solo_parts: Optional[List] = None

    def setup_scores(self):
        """
        Setup the score objects.

        This method initializes arguments used in the accompanion Base Class.
        This is called in the constructor.
        """
        tempo_model_type = self.tempo_model_kwargs.pop("tempo_model")

        if isinstance(tempo_model_type, str):
            tempo_model_type = getattr(tempo_models, tempo_model_type)

        self.tempo_model = tempo_model_type(**self.tempo_model_kwargs)

        self.solo_parts = []
        for i, fn in enumerate(self.score_kwargs["solo_fn"]):
            fn_ext = os.path.splitext(fn)[-1]
            if fn_ext == ".match":
                if i == 0:
                    solo_perf, alignment, solo_score = pt.load_match(
                        filename=fn,
                        create_score=True,
                        first_note_at_zero=True,
                    )
                    solo_ppart = solo_perf[0]
                    solo_spart = solo_score[0]
                else:
                    solo_perf, alignment = pt.load_match(
                        filename=fn,
                        create_score=False,
                        first_note_at_zero=True,
                    )
                    solo_ppart = solo_perf[0]

                ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
                    ppart_or_note_array=solo_ppart,
                    spart_or_note_array=solo_spart,
                    alignment=alignment,
                )
                self.solo_parts.append(
                    (solo_ppart, ptime_to_stime_map, stime_to_ptime_map)
                )
            else:
                # if the score is a score format supported by partitura
                # it will only load the first one
                if i == 0:
                    solo_spart = pt.load_score(fn)[0]

                    solo_pna, alignment = performance_notearray_from_score_notearray(
                        snote_array=solo_spart.note_array(),
                        bpm=self.init_bpm,
                        return_alignment=True,
                    )

                    solo_ppart = PerformedPart.from_note_array(solo_pna)

                    (
                        ptime_to_stime_map,
                        stime_to_ptime_map,
                    ) = get_time_maps_from_alignment(
                        ppart_or_note_array=solo_pna,
                        spart_or_note_array=solo_spart,
                        alignment=alignment,
                    )

                    self.solo_parts.append(
                        (solo_ppart, ptime_to_stime_map, stime_to_ptime_map)
                    )

        self.solo_score = part_to_score(solo_spart, bpm=self.init_bpm)

        if self.score_kwargs["accompaniment_match"] is None:
            acc_spart = pt.load_score(self.score_kwargs["acc_fn"])[0]
            acc_notes = list(part_to_score(acc_spart, bpm=self.init_bpm).notes)
            velocity_trend = None
            velocity_dev = None
            timing = None
            log_articulation = None
            log_bpr = None

        else:
            acc_perf, acc_alignment, acc_score = pt.load_match(
                filename=self.score_kwargs["accompaniment_match"],
                first_note_at_zero=True,
                create_score=True,
            )
            acc_ppart = acc_perf[0]
            acc_spart = acc_score[0]

            acc_pnote_array = acc_ppart.note_array()
            acc_snote_array = acc_spart.note_array()

            acc_notes = list(
                alignment_to_score(
                    fn_or_spart=acc_spart, ppart=acc_ppart, alignment=acc_alignment
                ).notes
            )
            pc = get_performance_codec(
                [
                    "velocity_trend",
                    "velocity_dev",
                    "beat_period",
                    "timing",
                    "articulation_log",
                ]
            )
            bm_params, _, u_onset_idx = pc.encode(
                part=acc_spart,
                ppart=acc_ppart,
                alignment=acc_alignment,
                return_u_onset_idx=True,
            )

            bm_params_onsetwise = notewise_to_onsetwise(bm_params, u_onset_idx)

            # TODO Use the solo part to compute the moving average
            vt_ma = moving_average_offline(
                parameter=bm_params_onsetwise["velocity_trend"],
                alpha=self.performance_codec_kwargs.get("velocity_trend_ma_alpha", 0.6),
            )

            velocity_trend = onsetwise_to_notewise(
                bm_params_onsetwise["velocity_trend"] / vt_ma, u_onset_idx
            )

            if self.tempo_model.has_tempo_expectations:
                # get iterable of the tempo expectations
                unique_acc_score_onsets = np.array(
                    [np.mean(acc_snote_array["onset_beat"][ui]) for ui in u_onset_idx]
                )
                self.tempo_model.tempo_expectations_func = interp1d(
                    x=unique_acc_score_onsets,
                    # np.unique(acc_spart.note_array()["onset_beat"]),
                    y=bm_params_onsetwise["beat_period"],
                    bounds_error=False,
                    kind="previous",
                    fill_value=(
                        bm_params_onsetwise["beat_period"][0],
                        bm_params_onsetwise["beat_period"][-1],
                    ),
                )
                self.init_bp = bm_params_onsetwise["beat_period"][0]
                self.beat_period = self.init_bp

            vd_scale = self.performance_codec_kwargs.get("velocity_dev_scale", 90)
            velocity_dev = bm_params["velocity_dev"] * vd_scale

            timing_scale = self.performance_codec_kwargs.get("timing_scale", 1.0)
            timing = bm_params["timing"] * timing_scale
            lart_scale = self.performance_codec_kwargs.get(
                "log_articulation_scale", 1.0
            )
            log_articulation = bm_params["articulation_log"] * lart_scale
            log_bpr = None

        self.acc_score = AccompanimentScore(
            notes=acc_notes,
            solo_score=self.solo_score,
            velocity_trend=velocity_trend,
            velocity_dev=velocity_dev,
            timing=timing,
            log_articulation=log_articulation,
            log_bpr=log_bpr,
        )

    def setup_score_follower(self):
        """
        Setup the score follower object.

        This method initializes arguments used in the accompanion Base Class.
        """
        input_pipeline_kwargs = self.score_follower_kwargs.pop("input_processor")
        input_processor_type = input_pipeline_kwargs.pop("processor")
        input_processor_kwargs = input_pipeline_kwargs.pop("processor_kwargs")
        score_follower_type = self.score_follower_kwargs.pop("score_follower")
        pipeline = SequentialOutputProcessor([PianoRollProcessor(piano_range=True)])

        state_to_ref_time_maps = []
        ref_to_state_time_maps = []
        score_followers = []

        for part, state_to_ref_time_map, ref_to_state_time_map in self.solo_parts:
            if state_to_ref_time_map is not None:
                ref_frames = partitura_to_framed_midi(
                    part_or_notearray_or_filename=part,
                    is_performance=True,
                    pipeline=pipeline,
                    polling_period=self.polling_period,
                )[0]

            else:
                raise NotImplementedError

            state_to_ref_time_maps.append(state_to_ref_time_map)
            ref_to_state_time_maps.append(ref_to_state_time_map)
            ref_features = np.array(ref_frames).astype(float)

            # setup score follower
            if score_follower_type == "OnlineTimeWarping":
                score_follower = OnlineTimeWarping(
                    reference_features=ref_features,
                    **self.score_follower_kwargs,
                )

            score_followers.append(score_follower)

        self.score_follower = MultiDTWScoreFollower(
            score_followers,
            state_to_ref_time_maps,
            ref_to_state_time_maps,
            self.polling_period,
        )

        if input_processor_type == "PianoRollProcessor":
            self.input_pipeline = SequentialOutputProcessor(
                [PianoRollProcessor(**input_processor_kwargs)]
            )
        else:
            raise NotImplementedError(f"Unknown input pipeline: {input_processor_type}")

    def check_empty_frames(self, frame):
        """
        Check if the frames are empty.

        Parameters
        ----------
        frame : np.ndarray
            The frame to check.
        Returns
        -------
        bool
        """
        if sum(frame) > 0:
            return False
        else:
            return True
