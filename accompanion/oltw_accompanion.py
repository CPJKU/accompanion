from typing import Optional, Iterable

import numpy as np
import partitura

from basismixer.performance_codec import get_performance_codec
from partitura.utils.music import get_time_maps_from_alignment
from basismixer.utils.music import onsetwise_to_notewise, notewise_to_onsetwise
from scipy.interpolate import interp1d

from accompanion.mtchmkr.alignment_online_oltw import (
    OnlineTimeWarping,
)
from accompanion.mtchmkr.utils_generic import SequentialOutputProcessor

from accompanion.base import ACCompanion
from accompanion.midi_handler.midi_input import POLLING_PERIOD

from accompanion.accompanist.score import (
    AccompanimentScore,
    alignment_to_score,
    part_to_score,
)
from accompanion.accompanist.accompaniment_decoder import (
    moving_average_offline,
)
from accompanion.mtchmkr.features_midi import PianoRollProcessor
from accompanion.utils.partitura_utils import (
    partitura_to_framed_midi_custom as partitura_to_framed_midi,
)
from accompanion.score_follower.trackers import MultiDTWScoreFollower
from accompanion.accompanist import tempo_models


class OLTWACCompanion(ACCompanion):
    def __init__(
        self,
        solo_fn,
        acc_fn,
        midi_router_kwargs: dict,  # this is just a workaround for now
        accompaniment_match: Optional[str] = None,
        midi_fn: Optional[str] = None,
        score_follower_kwargs: dict = {
            "score_follower": "OnlineTimeWarping",
            "window_size": 80,
            "step_size": 10,
            "input_processor": {
                "processor": "PianoRollProcessor",
                "processor_kwargs": {"piano_range": True},
            },
        },
        tempo_model_kwargs={"tempo_model": tempo_models.LSM},
        performance_codec_kwargs={
            "velocity_trend_ma_alpha": 0.6,
            "articulation_ma_alpha": 0.4,
            "velocity_dev_scale": 70,
            "velocity_min": 20,
            "velocity_max": 100,
            "velocity_solo_scale": 0.85,
            "timing_scale": 0.001,
            "log_articulation_scale": 0.1,
            "mechanical_delay": 0.0,
        },
        init_bpm: float = 60,
        init_velocity: int = 60,
        polling_period: float = POLLING_PERIOD,
        use_ceus_mediator: bool = False,
        adjust_following_rate: float = 0.1,
        bypass_audio: bool = False,  # bypass fluidsynth audio
        test: bool = False  # bypass MIDIRouter
    ) -> None:

        score_kwargs = dict(
            solo_fn=solo_fn if isinstance(solo_fn, Iterable) else [solo_fn],
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
            bypass_audio=bypass_audio,
            tempo_model_kwargs=tempo_model_kwargs,
            test=test
        )

        self.solo_parts = None

    def setup_scores(self):

        tempo_model_type = self.tempo_model_kwargs.pop("tempo_model")

        if isinstance(tempo_model_type, str):
            tempo_model_type = getattr(tempo_models, tempo_model_type)

        self.tempo_model = tempo_model_type(**self.tempo_model_kwargs)

        self.solo_parts = []
        for i, fn in enumerate(self.score_kwargs["solo_fn"]):
            if fn.endswith(".match"):
                if i == 0:
                    solo_ppart, alignment, solo_spart = partitura.load_match(
                        fn=fn, create_part=True, first_note_at_zero=True
                    )
                else:
                    solo_ppart, alignment = partitura.load_match(
                        fn=fn, create_part=False, first_note_at_zero=True
                    )

                ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
                    ppart_or_note_array=solo_ppart,
                    spart_or_note_array=solo_spart,
                    alignment=alignment,
                )
                self.solo_parts.append(
                    (solo_ppart, ptime_to_stime_map, stime_to_ptime_map)
                )
            else:
                solo_spart = partitura.load_score(fn)

                if i == 0:
                    solo_spart = solo_spart

                self.solo_parts.append((solo_spart, None, None))
        
        self.solo_score = part_to_score(solo_spart, bpm=self.init_bpm)
        # Dirty fix for partitura >= 1.1.0
        # print(type(solo_spart)) # partitura.score.Score
        # self.solo_score = solo_spart #bpm=self.init_bpm
        # print(type(solo_spart.parts[0])) # <class 'partitura.score.Part'>
        # self.solo_spart = self.solo_score.parts[0]

        if self.score_kwargs["accompaniment_match"] is None:
            acc_spart = partitura.load_score(self.score_kwargs["acc_fn"])
            acc_notes = list(part_to_score(acc_spart, bpm=self.init_bpm).notes)
            velocity_trend = None
            velocity_dev = None
            timing = None
            log_articulation = None
            log_bpr = None

        else:
            acc_ppart, acc_alignment, acc_spart = partitura.load_match(
                fn=self.score_kwargs["accompaniment_match"],
                first_note_at_zero=True,
                create_part=True,
            )
            # Dirty fix for partitura >= 1.1.0
            # print(type(acc_ppart), type(acc_alignment), type(acc_spart))
            # <class 'partitura.performance.Performance'> <class 'list'> <class 'partitura.score.Score'>
            # acc_ppart = acc_ppart.performedparts[0]
            # acc_spart = acc_spart.parts[0]
            # print(type(acc_ppart), type(acc_alignment), type(acc_spart))
            # <class 'partitura.performance.PerformedPart'> <class 'list'> <class 'partitura.score.Part'>
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
                self.tempo_model.tempo_expectations_func = interp1d(
                    np.unique(acc_spart.note_array()["onset_beat"]),
                    bm_params_onsetwise["beat_period"],
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

        pipeline_kwargs = self.score_follower_kwargs.pop("input_processor")
        score_follower_type = self.score_follower_kwargs.pop("score_follower")
        pipeline = SequentialOutputProcessor([PianoRollProcessor(piano_range=True)])

        state_to_ref_time_maps = []
        ref_to_state_time_maps = []
        score_followers = []

        # # reference score for visualization
        # self.reference_features = None

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

            # if self.reference_features is None:
            #     self.reference_features = ref_features

            # setup score follower
            # print(self.score_follower_kwargs)
            # print(self.score_follower_kwargs["score_follower"])
            score_follower = OnlineTimeWarping(
                reference_features=ref_features, **self.score_follower_kwargs
            )

            score_followers.append(score_follower)

        self.score_follower = MultiDTWScoreFollower(
            score_followers,
            state_to_ref_time_maps,
            ref_to_state_time_maps,
            self.polling_period,
        )

        self.input_pipeline = SequentialOutputProcessor(
            [PianoRollProcessor(piano_range=True)]
        )

    def check_empty_frames(self, frame):
        if sum(frame) > 0:
            return False
        else:
            return True
