# -*- coding: utf-8 -*-
"""
New utilities to be added to partitura!

TODO
----
* Replace utilities with the latest version from Partitura
"""

import mido
import partitura
import numpy as np
from basismixer.performance_codec import get_performance_codec
from basismixer.utils import get_unique_onset_idxs, notewise_to_onsetwise
from partitura import load_score, load_performance
from partitura.utils.music import performance_from_part
from accompanion.config import CONFIG
from partitura.score import Part
from partitura.performance import PerformedPart
from scipy.interpolate import interp1d


PPART_FIELDS = [
    ("onset_sec", "f4"),
    ("duration_sec", "f4"),
    ("pitch", "i4"),
    ("velocity", "i4"),
    ("track", "i4"),
    ("channel", "i4"),
    ("id", "U256"),
]


def dummy_pipeline(inputs):
    return inputs


def midi_messages_to_framed_midi(midi_msgs, msg_times, polling_period, pipeline):
    """
    Convert a list of MIDI messages to a framed MIDI representation
    Parameters
    ----------
    midi_msgs: list of mido.Message
        List of MIDI messages
    msg_times: list of float
        List of times (in seconds) at which the MIDI messages were received
    polling_period:
        Polling period (in seconds) used to convert the MIDI messages
    pipeline: function
        Function to be applied to the MIDI messages before converting them to a MIDI frame.

    Returns
    -------
    frames: list
        List of MIDI frames.
    """
    n_frames = int(np.ceil(msg_times.max() / polling_period))
    frame_times = (np.arange(n_frames) + 0.5) * polling_period

    frames = []
    for cursor in range(n_frames):

        if cursor == 0:
            # do not leave messages starting at 0 behind!
            idxs = np.where(msg_times <= polling_period)[0]
        else:
            idxs = np.where(
                np.logical_and(
                    msg_times > cursor * polling_period,
                    msg_times <= (cursor + 1) * polling_period,
                )
            )[0]

        output = pipeline(
            (list(zip(midi_msgs[idxs], msg_times[idxs])), frame_times[cursor])
        )
        frames.append(output)
    return frames


def get_time_maps_from_alignment(
    ppart_or_note_array, spart_or_note_array, alignment, remove_ornaments=True
):

    perf_note_array = partitura.utils.ensure_notearray(ppart_or_note_array)
    score_note_array = partitura.utils.ensure_notearray(spart_or_note_array)

    match_idx = get_matched_notes(score_note_array, perf_note_array, alignment)

    score_onsets = score_note_array[match_idx[:, 0]]["onset_beat"]
    score_durations = score_note_array[match_idx[:, 0]]["duration_beat"]

    perf_onsets = perf_note_array[match_idx[:, 1]]["onset_sec"]

    score_unique_onsets = np.unique(score_onsets)

    if remove_ornaments:
        # TODO: check that all onsets have a duration?
        # ornaments (grace notes) do not have a duration
        score_unique_onset_idxs = np.array(
            [
                np.where(np.logical_and(score_onsets == u, score_durations > 0))[0]
                for u in score_unique_onsets
            ],
            dtype=object,
        )

    else:
        score_unique_onset_idxs = np.array(
            [np.where(score_onsets == u)[0] for u in score_unique_onsets], dtype=object
        )

    eq_perf_onsets = np.array(
        [np.mean(perf_onsets[u]) for u in score_unique_onset_idxs]
    )

    ptime_to_stime_map = interp1d(
        x=eq_perf_onsets,
        y=score_unique_onsets,
        bounds_error=False,
        fill_value="extrapolate",
    )
    stime_to_ptime_map = interp1d(
        y=eq_perf_onsets,
        x=score_unique_onsets,
        bounds_error=False,
        fill_value="extrapolate",
    )

    return ptime_to_stime_map, stime_to_ptime_map


def expand_ornaments(part):
    """
    Modify the note_array of a Part to give an onset and duration to
    grace notes and rolled chords

    TODO
    """
    pass


def get_matched_notes(spart_note_array, ppart_note_array, gt_alignment):

    # Get matched notes
    matched_idxs = []
    for al in gt_alignment:
        # Get only matched notes (i.e., ignore inserted or deleted notes)
        if al["label"] == "match":

            # if ppart_note_array['id'].dtype != type(al['performance_id']):
            if not isinstance(ppart_note_array["id"], type(al["performance_id"])):
                p_id = str(al["performance_id"])
            else:
                p_id = al["performance_id"]

            p_idx = int(np.where(ppart_note_array["id"] == p_id)[0])

            s_idx = np.where(spart_note_array["id"] == al["score_id"])[0]

            if len(s_idx) > 0:
                s_idx = int(s_idx)
                matched_idxs.append((s_idx, p_idx))

    return np.array(matched_idxs)


def partitura_to_framed_midi_custom(
    part_or_notearray_or_filename,
    polling_period=CONFIG["POLLING_PERIOD"],
    pipeline=dummy_pipeline,
    is_performance=False,
    tempo_curve=None,
    score_bpm=100,
    return_reference=False,
):
    # Allow for loading all valid representations in partitura
    if isinstance(
        part_or_notearray_or_filename,
        (partitura.score.Part, partitura.performance.PerformedPart),
    ):
        reference = part_or_notearray_or_filename
    elif isinstance(part_or_notearray_or_filename, np.ndarray):
        # ensure that note array is a structured note array
        if part_or_notearray_or_filename.dtype.fields is not None:
            reference = part_or_notearray_or_filename
        else:
            raise ValueError("Input array is not a structured array!")
    else:
        if is_performance:
            reference = load_performance(part_or_notearray_or_filename)
        else:
            reference = load_score(part_or_notearray_or_filename)
    # Use this for now to compute Piano rolls for references.
    ref_notearray = partitura.utils.ensure_notearray(reference)

    onset_u, duration_u = partitura.utils.get_time_units_from_note_array(ref_notearray)
    min_ref_time = ref_notearray[onset_u].min()
    max_ref_time = (ref_notearray[onset_u] + ref_notearray[duration_u]).max()
    if is_performance:
        note_ons = ref_notearray[onset_u]
        note_offs = note_ons + ref_notearray[duration_u]
    else:
        # TODO @Carlos please verify this part
        unique_onsets = np.unique(ref_notearray[onset_u])
        if tempo_curve is None:
            default_bp = 60 / score_bpm
            tempo_curve = interp1d(
                unique_onsets,
                [default_bp] * len(unique_onsets),
                bounds_error=False,
                kind="previous",
                fill_value=(default_bp, default_bp),
            )

        # compute note_ons with respect to the given tempo curve
        bp = tempo_curve(unique_onsets).astype(np.float32)

        iois = np.diff(unique_onsets) * bp[:-1]
        unique_note_ons = np.r_[0, np.cumsum(iois)]

        onset_to_note_on_map = interp1d(
            unique_onsets, unique_note_ons, bounds_error="extrapolate"
        )
        note_ons = onset_to_note_on_map(ref_notearray[onset_u])
        note_offs = note_ons + ref_notearray[duration_u] * tempo_curve(
            ref_notearray[onset_u]
        ).astype(np.float32)

    midi_messages = []
    message_times = []

    onsets = {}
    for i, (non, noff, pitch) in enumerate(
        zip(note_ons, note_offs, ref_notearray["pitch"])
    ):

        if is_performance:
            velocity = ref_notearray[i]["velocity"]
        else:
            velocity = 64
        note_on = mido.Message("note_on", note=pitch, velocity=velocity)
        note_off = mido.Message("note_off", note=pitch, velocity=velocity)

        midi_messages += [note_on, note_off]
        message_times += [non, noff]

        frame = int(non / polling_period)

        if frame not in onsets:
            onsets[frame] = []

        onsets[frame].append(pitch - 21)

    # TODO: Add controls messages for performed parts
    if isinstance(reference, partitura.performance.PerformedPart):
        # for ctrl in reference.controls:
        #     # Do something
        pass

    midi_messages = np.array(midi_messages)
    message_times = np.array(message_times)
    sort_idx = np.argsort(message_times)
    midi_messages = midi_messages[sort_idx]
    message_times = message_times[sort_idx]

    frames = midi_messages_to_framed_midi(
        midi_messages, message_times, polling_period, pipeline
    )
    frames = decay_midi(np.asarray(frames).T, onsets).T
    # frame_times = np.arange(len(frames)) * polling_period
    ref_times = np.linspace(min_ref_time, max_ref_time, len(frames))
    state_to_ref_time_map = interp1d(
        np.arange(len(frames)),
        ref_times,
        bounds_error=False,
        fill_value=(min_ref_time, max_ref_time),
    )

    ref_time_to_state_map = interp1d(
        ref_times,
        np.arange(len(frames)),
        bounds_error=False,
        fill_value=(0, len(frames)),
    )

    if hasattr(pipeline, "reset"):
        # Reset pipeline (avoid carrying internal states for processing the
        # performance)
        pipeline.reset()

    output = (frames, note_ons, state_to_ref_time_map, ref_time_to_state_map)

    if return_reference:
        output += (reference,)

    return output


def decay_midi(frames, onsets):
    decay = np.ones(frames.shape[0])

    for i in range(frames.shape[1]):

        decay *= CONFIG["DECAY_VALUE"]
        if i in onsets:

            for o in onsets[i]:
                decay[o] = 1.0

        frames[:, i] *= decay
    return frames


def match2tempo(match_path):
    """
    computes tempo curve from a match file.

    Parameters
    ----------
    match_path : string
        a path of a match file

    Returns
    -------
    tempo_curve_and_onsets : numpy array
        score onsets and corresponding tempo curve stacked as columns
    """

    ppart, alignment, part = partitura.load_match(match_path, create_part=True)
    all_targets = list(set(["beat_period"]))
    perf_codec = get_performance_codec(all_targets)
    ppart.sustain_pedal_threshold = 128
    targets, snote_ids, ux = perf_codec.encode(
        part, ppart, alignment, return_u_onset_idx=True
    )

    nid_dict = dict((n.id, i) for i, n in enumerate(part.notes_tied))
    matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])

    score_onsets = part.note_array[matched_subset_idxs][
        "onset_beat"
    ]  # changed from onset_beat
    unique_onset_idxs, uni = get_unique_onset_idxs(
        score_onsets, return_unique_onsets=True
    )

    tmp_crv = notewise_to_onsetwise(
        np.array([targets["beat_period"]]).T, unique_onset_idxs
    )[:, 0]

    tempo_curve = interp1d(
        uni,
        tmp_crv,
        bounds_error=False,
        kind="previous",
        fill_value=(tmp_crv[0], tmp_crv[-1]),
    )

    return tempo_curve


def quarter_to_beat(note_duration, beat_type):
    factor = beat_type / 4.0
    return note_duration * factor


def beat_to_quarter(note_duration, beat_type):
    factor = beat_type / 4.0
    return note_duration / factor


def get_beat_conversion(note_duration, beat_type):

    from partitura.utils.music import DOT_MULTIPLIERS, LABEL_DURS

    dots = note_duration.count(".")
    unit = note_duration.strip().rstrip(".")
    duration_quarters = float(DOT_MULTIPLIERS[dots] * LABEL_DURS[unit])

    return quarter_to_beat(duration_quarters, beat_type)


def quarter_to_beat(note_duration, beat_type):
    factor = beat_type / 4.0
    return note_duration * factor


def beat_to_quarter(note_duration, beat_type):
    factor = beat_type / 4.0
    return note_duration / factor


def get_beat_conversion(note_duration, beat_type):

    from partitura.utils.music import DOT_MULTIPLIERS, LABEL_DURS

    dots = note_duration.count(".")
    unit = note_duration.strip().rstrip(".")
    duration_quarters = float(DOT_MULTIPLIERS[dots] * LABEL_DURS[unit])

    return quarter_to_beat(duration_quarters, beat_type)


if __name__ == "__main__":

    fn = "../demo_data/twinkle_twinkle_little_star_score.musicxml"

    spart = partitura.load_musicxml(fn)

    ppart = performance_from_part(spart)

    partitura.save_performance_midi(ppart, "../demo_data/test_pfp.mid")
