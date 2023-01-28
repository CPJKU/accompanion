# -*- coding: utf-8 -*-
import argparse
import os
import glob

from typing import Union, List, Tuple, Callable, Iterable, Any, Optional

import numpy as np
import partitura as pt
import matplotlib.pyplot as plt

from scipy.stats import skew, skewtest

from partitura.performance import Performance, PerformedPart, PerformanceLike
from partitura.score import Score, Part, ScoreLike
from partitura.utils.misc import PathLike

from partitura.utils.music import get_time_maps_from_alignment, compute_pianoroll

from accompanion.mtchmkr.alignment_online_oltw import OnlineTimeWarping

from accompanion.score_follower.trackers import (
    MultiDTWScoreFollower,
    AccompanimentScoreFollower,
)

from accompanion.utils.partitura_utils import (
    partitura_to_framed_midi_custom as partitura_to_framed_midi,
)

from accompanion.score_follower.onset_tracker import OnsetTracker

from accompanion.mtchmkr.utils_generic import SequentialOutputProcessor
from accompanion.mtchmkr.features_midi import PianoRollProcessor, PitchIOIProcessor
from accompanion.mtchmkr import score_hmm
from accompanion.score_follower.trackers import HMMScoreFollower
from accompanion.accompanist import tempo_models


PIANO_ROLL_PIPELINE = SequentialOutputProcessor([PianoRollProcessor(piano_range=True)])
DEFAULT_LOCAL_COST = "Manhattan"
WINDOW_SIZE = 100
STEP_SIZE = 5
START_WINDOW_SIZE = 60
POLLING_PERIOD = 0.01


def setup_scores(
    solo_fn: List[PathLike],
) -> Tuple[List[Tuple[PerformedPart, Callable, Callable]], Score]:
    """
    Setup the score objects.
    """
    solo_parts = []
    for i, fn in enumerate(solo_fn):
        if fn.endswith(".match"):
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
            solo_parts.append((solo_ppart, ptime_to_stime_map, stime_to_ptime_map))
        else:
            solo_spart = pt.load_score(fn)[0]

            if i == 0:
                solo_spart = solo_spart

            solo_parts.append((solo_spart, None, None))

    return solo_parts, solo_score


def compute_pianoroll_features(
    note_info: Union[ScoreLike, PerformanceLike],
    polling_period: float,
    pianoroll_method: str = "partitura",
):
    if pianoroll_method == "accompanion":
        ref_frames = partitura_to_framed_midi(
            part_or_notearray_or_filename=note_info,
            is_performance=True,
            pipeline=PIANO_ROLL_PIPELINE,
            polling_period=polling_period,
        )[0]

    elif pianoroll_method == "partitura":

        if isinstance(note_info, (Score, Part)):
            ppart = pt.utils.music.performance_from_part(note_info, bpm=60)
        else:
            ppart = note_info
        # Use real note off instead of sound off
        ppart.sustain_pedal_threshold = 127
        ref_frames = (
            compute_pianoroll(
                note_info=ppart,
                time_unit="sec",
                time_div=int(np.round(1 / polling_period)),
                binary=True,
                piano_range=True,
            )
            .toarray()
            .T
        )

    return ref_frames


def compute_hmm_features(
    note_info: Union[ScoreLike, PerformanceLike],
    polling_period: float = POLLING_PERIOD,
):

    if isinstance(note_info, (Score, Part)):
        ppart = pt.utils.music.performance_from_part(note_info, bpm=60)
    else:
        ppart = note_info

    features = partitura_to_framed_midi(
        part_or_notearray_or_filename=ppart,
        is_performance=True,
        pipeline=SequentialOutputProcessor([PitchIOIProcessor()]),
        polling_period=polling_period,
    )[0]

    return features


def setup_oltw(
    solo_parts,
    polling_period=POLLING_PERIOD,
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
    start_window_size=START_WINDOW_SIZE,
    local_cost_fun=DEFAULT_LOCAL_COST,
    pianoroll_method="partitura",
) -> AccompanimentScoreFollower:

    """
    Setup an OLTW score follower
    """
    # NOTE: pipeline_kwargs and score_follower_type are not used.
    state_to_ref_time_maps = []
    ref_to_state_time_maps = []
    score_followers = []

    for part, state_to_ref_time_map, ref_to_state_time_map in solo_parts:

        if state_to_ref_time_map is not None:
            ref_frames = compute_pianoroll_features(
                note_info=part,
                polling_period=polling_period,
                pianoroll_method=pianoroll_method,
            )

        state_to_ref_time_maps.append(state_to_ref_time_map)
        ref_to_state_time_maps.append(ref_to_state_time_map)
        ref_features = np.array(ref_frames).astype(float)

        # setup score follower
        score_follower = OnlineTimeWarping(
            reference_features=ref_features,
            window_size=window_size,
            step_size=step_size,
            local_cost_fun=local_cost_fun,
            start_window_size=start_window_size,
        )

        score_followers.append(score_follower)

    score_follower = MultiDTWScoreFollower(
        score_followers,
        state_to_ref_time_maps,
        ref_to_state_time_maps,
        polling_period,
    )

    return score_follower


def setup_hmm(
    solo_score_notearray: np.ndarray,
    ioi_precision: float = 2,
    gumbel_transition_matrix_scale: float = 0.5,
    init_bp: float = 0.5,
    trans_par: float = 1,
    trans_var: float = 0.03,
    obs_var: float = 0.0213,
    init_var: float = 1,
) -> AccompanimentScoreFollower:
    """
    Setup an HMM score follower
    """
    piano_range = False
    inserted_states = True

    unique_onsets = np.unique(solo_score_notearray["onset_beat"])
    unique_onset_idxs = [
        np.where(solo_score_notearray["onset_beat"] == ui)[0] for ui in unique_onsets
    ]

    chord_pitches = [solo_score_notearray["pitch"][ui] for ui in unique_onset_idxs]
    pitch_profiles = score_hmm.compute_pitch_profiles(
        chord_pitches,
        piano_range=piano_range,
        inserted_states=inserted_states,
    )
    ioi_matrix = score_hmm.compute_ioi_matrix(
        unique_onsets=unique_onsets,
        inserted_states=inserted_states,
    )
    state_space = ioi_matrix[0]
    n_states = len(state_space)

    transition_matrix = score_hmm.gumbel_transition_matrix(
        n_states=n_states,
        inserted_states=inserted_states,
        scale=gumbel_transition_matrix_scale,
    )
    initial_probabilities = score_hmm.gumbel_init_dist(n_states=n_states)

    score_follower = score_hmm.PitchIOIKHMM(
        transition_matrix=transition_matrix,
        pitch_profiles=pitch_profiles,
        ioi_matrix=ioi_matrix,
        score_onsets=state_space,
        init_beat_period=init_bp,
        init_score_onset=unique_onsets.min(),
        ioi_precision=ioi_precision,
        initial_probabilities=initial_probabilities,
        trans_par=trans_par,
        trans_var=trans_var,
        obs_var=obs_var,
        init_var=init_var,
    )

    score_follower = HMMScoreFollower(score_follower)

    return score_follower


def get_score_onset(
    frame,
    score_follower: AccompanimentScoreFollower,
    onset_tracker: OnsetTracker,
) -> float:

    score_position = score_follower(frame)
    solo_onset, _, _ = onset_tracker(score_position)

    if solo_onset is not None:
        return solo_onset


def compute_alignment(
    input_signal: Iterable,
    score_follower: AccompanimentScoreFollower,
    onset_tracker: OnsetTracker,
    polling_period: float = POLLING_PERIOD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the alignment
    """
    sf_output = [
        (get_score_onset(frame, score_follower, onset_tracker), i)
        for i, frame in enumerate(input_signal)
    ]

    sf_output = np.array([sfo for sfo in sf_output if sfo[0] is not None])

    tracked_sonsets = sf_output[:, 0]
    frames_in_input = sf_output[:, 1].astype(int)
    # tracked onset time correspond to the end of the frame
    tracked_ponsets = (sf_output[:, 1] + 1) * polling_period

    return tracked_sonsets, frames_in_input, tracked_ponsets


def evaluate_alignment(target_ponsets, tracked_ponsets):
    asynchrony = target_ponsets - tracked_ponsets
    abs_asynch = abs(asynchrony)
    mean_asynch = np.mean(abs_asynch)
    skewness = skew(asynchrony)
    sktest = skewtest(asynchrony, alternative="two-sided")
    lt_25ms = np.mean(abs_asynch <= 0.025)
    lt_50ms = np.mean(abs_asynch <= 0.05)
    lt_100ms = np.mean(abs_asynch <= 0.1)
    return mean_asynch, lt_25ms, lt_50ms, lt_100ms, skewness, skewtest


def alignment_experiment(
    solo_perf_fn: PathLike,
    reference_fn: Union[PathLike, List[PathLike]],
    follower_type: str,
    score_follower_kwargs: dict,
    out_dir: Optional[PathLike] = None,
    make_plots: bool = False,
):

    # Load performance of the solo(as a PerformedPart)
    solo_perf, _ = setup_scores([solo_perf_fn])

    # Load references
    solo_parts, solo_score = setup_scores(
        reference_fn if isinstance(reference_fn, (list, tuple)) else [reference_fn]
    )

    solo_score_notearray = solo_score.note_array()

    unique_onsets = np.unique(solo_score_notearray["onset_beat"])

    # Initialize onset tracker
    onset_tracker = OnsetTracker(unique_onsets=unique_onsets)

    if follower_type == "oltw":

        score_follower = setup_oltw(
            solo_parts=solo_parts,
            polling_period=POLLING_PERIOD,
            **score_follower_kwargs,
        )

        input_signal = compute_pianoroll_features(
            solo_perf[0][0],
            polling_period=POLLING_PERIOD,
            pianoroll_method="partitura",
        ).astype(float)

    elif follower_type == "hmm":

        score_follower = setup_hmm(
            solo_score_notearray=solo_score_notearray,
            **score_follower_kwargs,
        )

        input_signal = compute_hmm_features(
            solo_perf[0][0],
            polling_period=POLLING_PERIOD,
        )

    tracked_sonsets, frames_in_input, tracked_ponsets = compute_alignment(
        input_signal, score_follower, onset_tracker
    )

    target_ponsets = solo_perf[0][2](tracked_sonsets)

    mean_asynch, lt_25ms, lt_50ms, lt_100ms, skewness, sktest = evaluate_alignment(
        target_ponsets=target_ponsets,
        tracked_ponsets=tracked_ponsets,
    )

    if out_dir is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        alignment_fn = os.path.join(out_dir, "alignment.csv")

        np.savetxt(
            alignment_fn,
            np.column_stack(
                (tracked_sonsets, tracked_ponsets, target_ponsets, frames_in_input),
                delimiter=",",
                header=(
                    "score_onset_beats,tracked_perf_onset_seconds,"
                    "target_perf_onset_seconds,frame_number"
                ),
            ),
        )

        results_fn = os.path.join(out_dir, "results.csv")
        with open(results_fn, "w") as f:

            f.write(
                "#mean_asynch_ms,leq_25ms_%,leq_50ms_%,leq_100ms_%,skewness,sk_statistic,sk_pvalue\n"
            )
            f.write(
                f"{mean_asynch * 1000:.2f},{lt_25ms * 100:.1f},"
                f"{lt_100ms * 100:.1f},{lt_100ms * 100:.1f},{skewness:.2f}"
                f"{sktest.statistic:.3f},{sktest.pvalue:.4f}"
            )

    if make_plots:
        n_markers = 20
        input_pianoroll = compute_pianoroll_features(
            solo_perf[0][0],
            polling_period=POLLING_PERIOD,
            pianoroll_method="partitura",
        ).astype(float)

        plt.imshow(
            input_pianoroll[: frames_in_input[n_markers + 1]].T,
            aspect="auto",
            origin="lower",
            cmap="gray",
            interpolation="nearest",
        )
        plt.xticks(
            np.arange(len(input_pianoroll[: frames_in_input[n_markers + 1]]), 10),
            np.arange(
                len(input_pianoroll[: frames_in_input[n_markers + 1]]) * POLLING_PERIOD
            ),
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Piano Key")
        for fp in frames_in_input[:n_markers]:
            plt.plot(np.ones(88) * fp, np.arange(88), c="red", alpha=0.5)

        plt.figsave(os.path.join(out_dir, "alignment_example.pdf"))


if __name__ == "__main__":

    import os
    import glob

    data_dir = "../accompanion_pieces/complex_pieces/brahms_data/match/cc_solo"

    solo_perf_fn = "../accompanion_pieces/complex_pieces/brahms_data/match/cc_solo/Brahms_Hungarian-Dance-5_Primo_2021-07-27.match"

    follower_type = "hmm"

    match_fn = glob.glob(os.path.join(data_dir, "*.match"))

    # data_dir = "/Users/carlos/Downloads/sf_exp_data/match_solo/chopin_op09_No1"

    solo_perf_fn = "/Users/carlos/Downloads/sf_exp_data/match/chopin_op09_No2.match"
    match_fn = [solo_perf_fn]

    solo_perf, _ = setup_scores([solo_perf_fn])

    solo_parts, solo_score = setup_scores(match_fn)

    solo_score_notearray = solo_score.note_array()

    unique_onsets = np.unique(solo_score_notearray["onset_beat"])

    onset_tracker = OnsetTracker(unique_onsets=unique_onsets)

    if follower_type == "oltw":

        score_follower = setup_oltw(
            solo_parts=solo_parts,
            polling_period=POLLING_PERIOD,
            window_size=100,
            step_size=5,
            start_window_size=50,
        )

        input_signal = compute_pianoroll_features(
            solo_perf[0][0],
            polling_period=POLLING_PERIOD,
            pianoroll_method="partitura",
        ).astype(float)

    elif follower_type == "hmm":

        score_follower = setup_hmm(
            solo_score_notearray=solo_score_notearray,
            ioi_precision=1,
            gumbel_transition_matrix_scale=0.5,
            init_bp=0.8,
            trans_par=1,
            trans_var=0.03,
            obs_var=0.0213,
            init_var=1,
        )

        input_signal = compute_hmm_features(
            solo_perf[0][0],
            polling_period=POLLING_PERIOD,
        )

    tracked_sonsets, frames_in_input, tracked_ponsets = compute_alignment(
        input_signal, score_follower, onset_tracker
    )

    target_ponsets = solo_perf[0][2](tracked_sonsets)

    evaluate_alignment(target_ponsets=target_ponsets, tracked_ponsets=tracked_ponsets)
    make_plots = True

    if make_plots:
        n_markers = 20
        input_pianoroll = compute_pianoroll_features(
            solo_perf[0][0],
            polling_period=POLLING_PERIOD,
            pianoroll_method="partitura",
        ).astype(float)

        plt.imshow(
            input_pianoroll[: frames_in_input[n_markers + 1]].T,
            aspect="auto",
            origin="lower",
            cmap="gray",
            interpolation="nearest",
        )
        for fp in frames_in_input[:n_markers]:
            plt.plot(np.ones(88) * fp, np.arange(88), c="red", alpha=0.5)

        plt.show()
