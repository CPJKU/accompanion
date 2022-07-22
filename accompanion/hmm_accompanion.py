from typing import Optional


import numpy as np
import partitura

from basismixer.performance_codec import get_performance_codec
from basismixer.utils.music import onsetwise_to_notewise, notewise_to_onsetwise
from scipy.interpolate import interp1d

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
from accompanion.mtchmkr.features_midi import PitchIOIProcessor
from accompanion.score_follower.trackers import HMMScoreFollower
from accompanion.accompanist import tempo_models
from accompanion.mtchmkr import score_hmm


class HMMACCompanion(ACCompanion):
    def __init__(
        self,
        solo_fn,
        acc_fn,
        midi_router_kwargs: dict,  # this is just a workaround for now
        accompaniment_match: Optional[str] = None,
        midi_fn: Optional[str] = None,
        score_follower_kwargs: dict = {
            "score_follower": "PitchIOIHMM",
            # For the future!
            "score_follower_kwargs": {},
            "input_processor": {
                "processor": "PitchIOIProcessor",
                "processor_kwargs": {},
            },
            # For the Future!
            # "reference_processor": {}
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
        test: bool = False # bypass MIDIRouter
    ) -> None:

        score_kwargs = dict(
            solo_fn=solo_fn,
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
            test=test,
            onset_tracker_type="continuous",
        )

    def setup_scores(self):

        tempo_model_type = self.tempo_model_kwargs.pop("tempo_model")

        if isinstance(tempo_model_type, str):
            tempo_model_type = getattr(tempo_models, tempo_model_type)

        self.tempo_model = tempo_model_type(**self.tempo_model_kwargs)

        solo_spart = partitura.load_score(self.score_kwargs["solo_fn"])

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

        self.solo_score = part_to_score(solo_spart, bpm=60 / self.init_bp, velocity=64)

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

        # These parameters should go in the score follower kwargs
        # but for now are here since there are no other alternatives
        piano_range = False
        inserted_states = True
        ioi_precision = 2
        pipeline_kwargs = self.score_follower_kwargs.pop("input_processor")
        score_follower_type = self.score_follower_kwargs.pop("score_follower")

        try:
            score_follower_kwargs = self.score_follower_kwargs.pop("score_follower_kwargs")
        except KeyError:
            score_follower_kwargs = {}

        # try:
        chord_pitches = [chord.pitch for chord in self.solo_score.chords]
        # except:
        #     print(self.solo_score.chords)
        pitch_profiles = score_hmm.compute_pitch_profiles(
            chord_pitches, piano_range=piano_range, inserted_states=inserted_states,
        )
        ioi_matrix = score_hmm.compute_ioi_matrix(
            unique_onsets=self.solo_score.unique_onsets, inserted_states=inserted_states,
        )
        state_space = ioi_matrix[0]
        n_states = len(state_space)
        transition_matrix = score_hmm.gumbel_transition_matrix(
            n_states=n_states,
            inserted_states=inserted_states,
            scale=0.5,
        )
        initial_probabilities = score_hmm.gumbel_init_dist(n_states=n_states)

        if score_follower_type == 'PitchIOIHMM':
            score_follower = score_hmm.PitchIOIHMM(
                transition_matrix=transition_matrix,
                pitch_profiles=pitch_profiles,
                ioi_matrix=ioi_matrix,
                score_onsets=state_space,
                tempo_model=self.tempo_model,
                ioi_precision=ioi_precision,
                initial_probabilities=initial_probabilities,
                **score_follower_kwargs,
            )
        elif score_follower_type == "PitchIOIKHMM":
            score_follower = score_hmm.PitchIOIKHMM(
                transition_matrix=transition_matrix,
                pitch_profiles=pitch_profiles,
                ioi_matrix=ioi_matrix,
                score_onsets=state_space,
                init_beat_period=self.init_bp,
                ioi_precision=ioi_precision,
                initial_probabilities=initial_probabilities,
                **score_follower_kwargs,
            )

        else:
            raise ValueError(f"{score_follower_type} is not a valid score HMM")
        self.score_follower = HMMScoreFollower(score_follower)
        self.input_pipeline = SequentialOutputProcessor([PitchIOIProcessor()])

    def check_empty_frames(self, frame):
        if frame is None:
            return True
        else:
            return False
