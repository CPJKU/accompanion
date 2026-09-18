#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tempo curves of recorded duo performances, for the virtual avatar.

For every recording in the duo dataset -- one MIDI file per player, of a piece
the ACCompanion has the score of -- this writes out the map between
performance time (seconds) and score time (beats), estimated two ways:

``online``
    What the ACCompanion estimates *while following* the performance: the
    onsets its score follower reports, the tempo model's beat period after
    each of them, and the dead-reckoned score position on every input frame.
    Produced by driving `ACCompanion.follow_step` -- the code that plays live
    -- over the MIDI file, so it is exactly the signal the avatar will get at
    run time: causal, smoothed, and occasionally wrong.

``offline``
    The best map that can be had in hindsight: a parangonar note alignment of
    the performance to the score, reduced to one performed time per score
    onset, with a local tempo fitted around each. Ground truth for normalising
    the training data, and the yardstick the online estimate is measured
    against.

Both are reported in the MIDI file's own clock, in seconds. The motion capture
is assumed to be on that clock too (pyanoduo rebases video time to
``frame_index / fps``, i.e. seconds from the first frame). To warp a motion
frame at ``t`` seconds into score time::

    from extract_tempo_curves import warp_table
    knots = np.load("D05/B1_T1_L1/p1.npz")["offline_knots"]
    beat = np.interp(t, *warp_table(knots))

or, for what the avatar will see live::

    frames = np.load(...)["online_frames"]
    beat = np.interp(t, frames["time_sec"], frames["expected_position_beat"])

The dataset layout is ``<root>/<duo>/<take>/midi_p1.mid`` and ``midi_p2.mid``.
The piece is read off the take's block (``B1_...`` is the badinerie,
``B2_...`` the fanfare, see ``--block-piece``), and which player is the primo
is worked out from the pitch content, since the players swap parts between
takes.

Usage
-----
    python bin/extract_tempo_curves.py --dataset ~/datasets/aura/midi_dataset --out tempo_curves
    python bin/extract_tempo_curves.py --dataset ... --takes 'D05/*' --methods offline
    python bin/extract_tempo_curves.py --midi take/midi_p1.mid --piece badinerie --role primo --out x

Output
------
``<out>/<duo>/<take>/<player>.npz`` holds, per player (``p1``, ``p2``):

``offline_knots``
    One row per score onset the alignment placed: ``score_onset_beat``,
    ``score_onset_quarter``, ``perf_onset_sec`` (the first matched note of the
    chord), ``beat_period`` and ``quarter_period`` (seconds per beat / per
    quarter note: the median over the knots within ``--window`` beats, holds
    left out), ``beat_period_raw`` (to the next knot only), ``n_notes``,
    ``is_fermata`` (the score puts a fermata here) and ``hold_sec`` (how much
    longer than the tempo explains the player sat on it). `warp_table` turns
    the knots into an `np.interp` table that keeps the beat still for the
    length of each hold.
``offline_alignment``
    parangonar's note-level alignment: ``label``, ``perf_id``, ``score_id``,
    ``perf_onset_sec``, ``score_onset_beat``.
``online_onsets``
    One row per onset the score follower reported: ``perf_onset_sec``,
    ``score_onset_beat``, ``score_onset_quarter``, ``beat_period`` (the tempo
    model's estimate right after this onset), ``quarter_period``, ``bpm``,
    ``est_onset_sec`` and ``asynchrony_sec`` (the tempo model's prediction of
    this onset and its error), ``tempo_updated`` (whether this onset reached
    the tempo model at all), ``expected_position_beat`` (the dead reckoning
    after its pull towards this onset), and ``accompanion_beat_period``
    (`ACCompanion.beat_period` as the running code reports it -- differs from
    ``beat_period`` on branches without the `pc.bp_ave` fix).
``online_frames``
    One row per input frame: ``time_sec``, ``expected_position_beat``,
    ``beat_period``, ``waiting`` (held at a fermata).
``online_alignment``
    The note tracker's alignment of performed notes to score ids.
``online_accompaniment``
    ``score_onset_beat`` and ``perf_onset_sec`` of every accompaniment onset
    the ACCompanion would have played, from a simulated sequencer.

``<out>/<duo>/<take>/summary.json`` carries the roles, the alignment counts,
and, where both maps exist, how far the online estimate is from the offline
one. ``<out>/index.json`` collects the summaries.

The online map is only produced for the player the ACCompanion follows: the
primo by default, or both players with ``--follow both`` (then the secondo is
followed as if it were the solo, with the primo as accompaniment).
"""
import argparse
import fnmatch
import glob
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "bin"))

import numpy as np
import partitura as pt
import yaml

import test_accompaniment as A
import test_followers as H
from accompanion.base import FollowingState
from accompanion.score_follower.onset_tracker import DiscreteOnsetTracker, OnsetTracker

#: Which piece each block of the dataset records.
DEFAULT_BLOCK_PIECES = {"B1": "badinerie", "B2": "fanfare"}

#: How long before the piece's first note the ACCompanion is started, in
#: seconds. The takes open with several seconds of silence, and the
#: ACCompanion dead reckons from the moment it is launched, so a soloist who
#: does not start at once drags the follower ahead of the music. Live, the
#: ACCompanion is launched right before playing; this reproduces that. The
#: first note is the first one the offline alignment placed when that map is
#: being made too, so that keys tried before the piece do not count.
LEAD_IN_SEC = 0.5

ROLES = ("primo", "secondo")

OFFLINE_KNOT_DTYPE = [
    ("score_onset_beat", "f8"),
    ("score_onset_quarter", "f8"),
    ("perf_onset_sec", "f8"),
    ("beat_period", "f8"),
    ("quarter_period", "f8"),
    ("beat_period_raw", "f8"),
    ("n_notes", "i4"),
    ("is_fermata", "?"),
    ("hold_sec", "f8"),
]
OFFLINE_ALIGNMENT_DTYPE = [
    ("label", "U9"),
    ("perf_id", "U16"),
    ("score_id", "U16"),
    ("perf_onset_sec", "f8"),
    ("score_onset_beat", "f8"),
]
ONLINE_ONSET_DTYPE = [
    ("perf_onset_sec", "f8"),
    ("score_onset_beat", "f8"),
    ("score_onset_quarter", "f8"),
    ("beat_period", "f8"),
    ("quarter_period", "f8"),
    ("bpm", "f8"),
    ("est_onset_sec", "f8"),
    ("asynchrony_sec", "f8"),
    ("tempo_updated", "?"),
    ("expected_position_beat", "f8"),
    ("accompanion_beat_period", "f8"),
]
ONLINE_FRAME_DTYPE = [
    ("time_sec", "f8"),
    ("expected_position_beat", "f8"),
    ("beat_period", "f8"),
    ("waiting", "?"),
]
ONLINE_ALIGNMENT_DTYPE = [
    ("label", "U9"),
    ("score_id", "U16"),
    ("perf_onset_sec", "f8"),
]
ONLINE_ACCOMPANIMENT_DTYPE = [
    ("score_onset_beat", "f8"),
    ("perf_onset_sec", "f8"),
]


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------
class Piece(object):
    """The two parts of a duo piece, and the beat <-> quarter maps of each."""

    def __init__(self, name):
        self.name = name
        self.files = dict(zip(ROLES, H.find_piece(name)))
        self.parts = {role: pt.load_score(fn)[0] for role, fn in self.files.items()}
        self.note_arrays = {
            role: part.note_array(include_grace_notes=True)
            for role, part in self.parts.items()
        }
        #: Score onsets carrying a fermata, in beats, per part.
        self.fermatas = {
            role: np.unique([
                float(part.beat_map(f.start.t))
                for f in part.iter_all(pt.score.Fermata)
                if f.start is not None
            ])
            for role, part in self.parts.items()
        }

    def to_quarter(self, role, beats):
        """Score onsets in beats -> quarter notes, for the part of `role`."""
        part = self.parts[role]
        beats = np.asarray(beats, dtype=float)
        if len(beats) == 0:
            return beats
        return np.asarray(part.quarter_map(part.inv_beat_map(beats)), dtype=float)

    def beats_per_quarter(self, role, beats):
        """How many beats a quarter note lasts at each of `beats`.

        1 in 4/4, 2 in 6/8, 0.5 in 2/2: the factor that turns a beat period
        into a quarter-note period across a change of time signature.
        """
        part = self.parts[role]
        beats = np.asarray(beats, dtype=float)
        if len(beats) == 0:
            return beats
        divs = np.asarray(part.inv_beat_map(beats), dtype=float)
        # One division either side, so that a knot sitting exactly on a
        # meter change takes the meter it starts.
        b0, b1 = part.beat_map(divs), part.beat_map(divs + 1)
        q0, q1 = part.quarter_map(divs), part.quarter_map(divs + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (np.asarray(b1) - np.asarray(b0)) / (np.asarray(q1) - np.asarray(q0))
        ratio = np.asarray(ratio, dtype=float)
        ratio[~np.isfinite(ratio)] = 1.0
        return ratio


# ---------------------------------------------------------------------------
# Which player plays which part
# ---------------------------------------------------------------------------
def pitch_similarity(pitches_a, pitches_b):
    """Cosine similarity of two pitch histograms."""
    ha = np.bincount(np.asarray(pitches_a, dtype=int), minlength=128).astype(float)
    hb = np.bincount(np.asarray(pitches_b, dtype=int), minlength=128).astype(float)
    denominator = np.linalg.norm(ha) * np.linalg.norm(hb)
    return float(ha @ hb / denominator) if denominator > 0 else 0.0


def assign_roles(performances, piece):
    """``{player: role}`` for the assignment whose pitch content fits best.

    Parameters
    ----------
    performances : dict
        ``{player: note array}`` for the two players.
    piece : Piece

    Returns
    -------
    roles : dict
    margin : float
        How much better the chosen assignment fits than the other one; near
        zero means the parts are indistinguishable by pitch and the choice
        should be checked.
    """
    players = sorted(performances)
    if len(players) != 2:
        raise ValueError(f"Expected two players, got {players}")

    def fit(assignment):
        return sum(
            pitch_similarity(
                performances[player]["pitch"], piece.note_arrays[role]["pitch"]
            )
            for player, role in assignment.items()
        )

    straight = {players[0]: "primo", players[1]: "secondo"}
    swapped = {players[0]: "secondo", players[1]: "primo"}
    s, w = fit(straight), fit(swapped)
    return (straight, s - w) if s >= w else (swapped, w - s)


# ---------------------------------------------------------------------------
# Offline: parangonar alignment -> beat map
# ---------------------------------------------------------------------------
def make_matcher(name):
    import parangonar

    if name == "dualdtw":
        return parangonar.DualDTWNoteMatcher()
    if name == "automatic":
        return parangonar.AutomaticNoteMatcher()
    raise ValueError(f"Unknown matcher '{name}'")


def longest_increasing(values):
    """Indices of a longest strictly increasing subsequence of `values`.

    Used to discard the matches that cannot all be right: a performance runs
    forward in time, so the performed times of successive score onsets must
    too, and the largest set of matches that agrees on that is kept.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([], dtype=int)
    tails, tails_idx, prev = [], [], np.full(n, -1)
    for i, v in enumerate(values):
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        if lo == len(tails):
            tails.append(v)
            tails_idx.append(i)
        else:
            tails[lo] = v
            tails_idx[lo] = i
        prev[i] = tails_idx[lo - 1] if lo > 0 else -1
    out, i = [], tails_idx[-1]
    while i >= 0:
        out.append(i)
        i = prev[i]
    return np.array(out[::-1], dtype=int)


def local_tempo(positions, raw, centers, window, exclude=None):
    """Median of `raw` over the knots within `window` beats of each center.

    A median rather than a fit, so that one held note among the knots in
    reach leaves the tempo alone; the knots in `exclude` (the fermatas,
    whose raw period *is* the hold) are left out altogether. Falls back to
    the knot's own raw period where nothing else is in reach.
    """
    positions, raw = np.asarray(positions, dtype=float), np.asarray(raw, dtype=float)
    usable = np.isfinite(raw)
    if exclude is not None:
        usable &= ~np.asarray(exclude, dtype=bool)
    out = np.array(raw, dtype=float)
    for k, c in enumerate(centers):
        mask = usable & (np.abs(positions - c) <= window)
        if mask.any():
            out[k] = np.median(raw[mask])
    return out


def warp_table(knots):
    """``(seconds, beats)`` to hand to `np.interp`, holds included.

    A fermata knot is followed by a second point ``hold_sec`` later at the
    same beat, so that the beat stands still for the length of the hold and
    then moves on to the next onset at the local tempo -- rather than the
    hold being spread thin over the whole interval to the next knot.
    """
    seconds, beats = [], []
    for knot in knots:
        seconds.append(knot["perf_onset_sec"])
        beats.append(knot["score_onset_beat"])
        if knot["is_fermata"] and knot["hold_sec"] > 0:
            seconds.append(knot["perf_onset_sec"] + knot["hold_sec"])
            beats.append(knot["score_onset_beat"])
    return np.asarray(seconds, dtype=float), np.asarray(beats, dtype=float)


def offline_map(piece, role, performed_part, matcher, window):
    """Align a performance to its part and reduce it to a beat map.

    Returns
    -------
    knots : np.ndarray (OFFLINE_KNOT_DTYPE)
    alignment : np.ndarray (OFFLINE_ALIGNMENT_DTYPE)
    raw_alignment : list of dict
        parangonar's output, for `partitura.save_match`.
    stats : dict
    """
    score_na = piece.note_arrays[role]
    perf_na = performed_part.note_array()
    raw_alignment = matcher(score_na, perf_na)

    score_index = {sid: i for i, sid in enumerate(score_na["id"])}
    perf_index = {pid: i for i, pid in enumerate(perf_na["id"])}

    rows, pairs = [], []
    for entry in raw_alignment:
        label = entry["label"]
        sid, pid = entry.get("score_id", ""), entry.get("performance_id", "")
        s = score_index.get(sid)
        p = perf_index.get(pid)
        score_onset = float(score_na["onset_beat"][s]) if s is not None else np.nan
        perf_onset = float(perf_na["onset_sec"][p]) if p is not None else np.nan
        rows.append((label, pid or "", sid or "", perf_onset, score_onset))
        # A grace note shares its onset with the note it decorates but is
        # played before it, so it does not say when the onset was.
        if label == "match" and s is not None and not score_na["is_grace"][s]:
            pairs.append((score_onset, perf_onset))
    alignment = np.array(rows, dtype=OFFLINE_ALIGNMENT_DTYPE)

    stats = {
        "matched": int(np.sum(alignment["label"] == "match")),
        "inserted": int(np.sum(alignment["label"] == "insertion")),
        "deleted": int(np.sum(alignment["label"] == "deletion")),
        "score_notes": int(len(score_na)),
        "performed_notes": int(len(perf_na)),
    }
    if not pairs:
        return np.array([], dtype=OFFLINE_KNOT_DTYPE), alignment, raw_alignment, stats

    pairs = np.array(pairs)
    # One knot per score onset: the chord is "on" when its first key goes
    # down, which is also the moment the online follower sees it.
    onsets = np.unique(pairs[:, 0])
    perf_times = np.array([pairs[pairs[:, 0] == o, 1].min() for o in onsets])
    n_notes = np.array([np.sum(pairs[:, 0] == o) for o in onsets])

    keep = longest_increasing(perf_times)
    stats["knots"] = int(len(keep))
    stats["knots_dropped"] = int(len(onsets) - len(keep))
    onsets, perf_times, n_notes = onsets[keep], perf_times[keep], n_notes[keep]

    quarters = piece.to_quarter(role, onsets)
    fermatas = piece.fermatas[role]
    is_fermata = (
        np.any(np.abs(onsets[:, None] - fermatas[None, :]) < 1e-6, axis=1)
        if len(fermatas)
        else np.zeros(len(onsets), dtype=bool)
    )
    raw = np.full(len(onsets), np.nan)
    if len(onsets) > 1:
        raw[:-1] = np.diff(perf_times) / np.diff(onsets)
        raw[-1] = raw[-2]
    beat_period = local_tempo(onsets, raw, onsets, window, exclude=is_fermata)
    quarter_period = beat_period * piece.beats_per_quarter(role, onsets)
    # How much longer than the tempo explains the player sat on each fermata.
    hold = np.zeros(len(onsets))
    if len(onsets) > 1:
        hold[:-1] = np.diff(onsets) * (raw[:-1] - beat_period[:-1])
    hold = np.where(is_fermata, np.maximum(hold, 0.0), 0.0)
    stats["fermatas_held"] = int(np.sum(hold > 0))
    stats["hold_sec_total"] = float(hold.sum())

    knots = np.zeros(len(onsets), dtype=OFFLINE_KNOT_DTYPE)
    knots["score_onset_beat"] = onsets
    knots["score_onset_quarter"] = quarters
    knots["perf_onset_sec"] = perf_times
    knots["beat_period"] = beat_period
    knots["quarter_period"] = quarter_period
    knots["beat_period_raw"] = raw
    knots["n_notes"] = n_notes
    knots["is_fermata"] = is_fermata
    knots["hold_sec"] = hold
    return knots, alignment, raw_alignment, stats


# ---------------------------------------------------------------------------
# Online: the ACCompanion following the performance
# ---------------------------------------------------------------------------
def load_config(name):
    """The ``config`` section of ``config_files/<name>.yml``, or None."""
    fn = os.path.join(REPO_ROOT, "config_files", name + ".yml")
    if not os.path.exists(fn):
        return None
    with open(fn, "rb") as f:
        return dict(yaml.safe_load(f)["config"])


def build_from_config(config, solo_fn, acc_fn, polling_period=None, init_bpm=None):
    """An ACCompanion as `launch_acc.py` would build it from a config file.

    Same class, same score follower, tempo model, performance codec and
    fermata settings; only the MIDI ports are left unopened.
    """
    config = dict(config)
    follower = config.pop("follower", "hmm")
    for key in ("midi_fn", "record_midi", "piece_dir"):
        config.pop(key, None)
    config["midi_router_kwargs"] = {
        key: None for key in config.get("midi_router_kwargs", {})
    }
    config.update(solo_fn=solo_fn, acc_fn=acc_fn, test=True, bypass_audio=True)
    if polling_period is not None:
        config["polling_period"] = polling_period
    if init_bpm is not None:
        config["init_bpm"] = init_bpm

    if follower == "hmm":
        from accompanion.hmm_accompanion import HMMACCompanion as cls
    elif follower == "oltw":
        from accompanion.oltw_accompanion import OLTWACCompanion as cls
    elif follower == "matchmaker":
        from accompanion.matchmaker_accompanion import MatchmakerACCompanion as cls
    else:
        raise ValueError(f"Unknown follower variant '{follower}' in config")

    accompanion = cls(**config)
    accompanion.setup_following()
    return accompanion


def build_accompanion(args, piece, solo_role):
    """The ACCompanion that follows `solo_role`, per the command line."""
    acc_role = "secondo" if solo_role == "primo" else "primo"
    solo_fn, acc_fn = piece.files[solo_role], piece.files[acc_role]

    if args.follower is not None:
        polling_period = H.polling_period_for(args.follower, args.polling_period)
        follower_kwargs = json.loads(args.follower_kwargs) if args.follower_kwargs else None
        return A.build(
            args.follower, solo_fn, acc_fn, polling_period,
            args.init_bpm if args.init_bpm is not None else 60.0,
            follower_kwargs=follower_kwargs,
        )

    config_name = args.config if args.config is not None else piece.name
    config = load_config(config_name)
    if config is None:
        raise SystemExit(
            f"No config_files/{config_name}.yml; pass --config or --follower."
        )
    return build_from_config(
        config, solo_fn, acc_fn,
        polling_period=args.polling_period, init_bpm=args.init_bpm,
    )


def online_map(
    accompanion, piece, solo_role, midi_fn, lead_in=LEAD_IN_SEC, first_note=None
):
    """Follow `midi_fn` with `accompanion` and record what it estimates.

    Parameters
    ----------
    first_note : float, optional
        When the piece starts, in the file's clock. The ACCompanion is
        launched `lead_in` seconds before it and hears nothing earlier -- a
        player who tries a few keys before starting would otherwise send the
        follower off before the music begins. Defaults to the first note in
        the file; the offline alignment knows better.

    Returns
    -------
    onsets : np.ndarray (ONLINE_ONSET_DTYPE)
    frames : np.ndarray (ONLINE_FRAME_DTYPE)
    alignment : np.ndarray (ONLINE_ALIGNMENT_DTYPE)
    accompaniment : np.ndarray (ONLINE_ACCOMPANIMENT_DTYPE)
    stats : dict
    """
    messages, times = H.messages_from_midi(midi_fn)
    if first_note is None:
        first_note = float(min(times))
    # Start the ACCompanion `lead_in` seconds before the piece, and report
    # everything in the file's own clock again afterwards.
    shift = float(first_note) - lead_in
    kept = [i for i, t in enumerate(times) if t >= shift]
    dropped = len(messages) - len(kept)
    messages = [messages[i] for i in kept]
    times = [times[i] - shift for i in kept]

    polling_period = accompanion.polling_period
    if getattr(accompanion, "event_based_input", False):
        input_frames = H.event_frames(messages, times)
    else:
        input_frames = H.framed(messages, times, polling_period)

    tracker_cls = (
        DiscreteOnsetTracker
        if accompanion.onset_tracker_type == "discrete"
        else OnsetTracker
    )
    onset_tracker = tracker_cls(accompanion.solo_score.unique_onsets)
    state = FollowingState(expected_position=accompanion.first_score_onset)
    state.solo_starts = (
        accompanion.acc_score.min_onset >= accompanion.solo_score.min_onset
    )
    sequencer = A.SimulatedSequencer(accompanion.acc_score)
    tempo_model = accompanion.tempo_model

    onset_rows, frame_rows = [], []
    started = time.perf_counter()
    with A.quiet():
        for frame, t in input_frames:
            n_before = len(accompanion.time_delays)
            calls_before = tempo_model.counter
            accompanion.follow_step(
                input_midi_messages=frame,
                output=accompanion.input_pipeline((frame, t)),
                solo_p_onset=t,
                onset_tracker=onset_tracker,
                state=state,
            )
            sequencer.step(t)

            waiting = bool(accompanion.fermata_hold.waiting)
            frame_rows.append(
                (t + shift, state.expected_position, tempo_model.beat_period, waiting)
            )
            if len(accompanion.time_delays) > n_before:
                score_onset, perf_onset, _ = accompanion.time_delays[-1]
                est = tempo_model.est_onset
                onset_rows.append((
                    perf_onset + shift,
                    score_onset,
                    np.nan,  # quarter, filled below
                    tempo_model.beat_period,
                    np.nan,  # quarter period, filled below
                    60.0 / tempo_model.beat_period,
                    (est + shift) if est is not None else np.nan,
                    float(tempo_model.asynchrony),
                    tempo_model.counter > calls_before,
                    state.expected_position,
                    accompanion.beat_period,
                ))
    elapsed = time.perf_counter() - started

    onsets = np.array(onset_rows, dtype=ONLINE_ONSET_DTYPE)
    if len(onsets):
        beats = onsets["score_onset_beat"]
        onsets["score_onset_quarter"] = piece.to_quarter(solo_role, beats)
        onsets["quarter_period"] = onsets["beat_period"] * piece.beats_per_quarter(
            solo_role, beats
        )
    frames = np.array(frame_rows, dtype=ONLINE_FRAME_DTYPE)

    alignment = np.array(
        [
            (a["label"], str(a.get("score_id", "")), float(a["onset"]) + shift)
            for a in accompanion.note_tracker.alignment
        ],
        dtype=ONLINE_ALIGNMENT_DTYPE,
    )

    played = {}
    for onset, when in sequencer.played:
        played.setdefault(float(onset), float(when) + shift)
    accompaniment = np.array(sorted(played.items()), dtype=ONLINE_ACCOMPANIMENT_DTYPE)

    solo_onsets = accompanion.solo_score.unique_onsets
    stats = {
        "reported_onsets": int(len(onsets)),
        "distinct_onsets": int(len(np.unique(onsets["score_onset_beat"]))) if len(onsets) else 0,
        "score_onsets": int(len(solo_onsets)),
        "tempo_updates": int(np.sum(onsets["tempo_updated"])) if len(onsets) else 0,
        "frames": int(len(frames)),
        "frames_waiting": int(np.sum(frames["waiting"])),
        "accompaniment_played": int(len(accompaniment)),
        "accompaniment_onsets": int(len(accompanion.acc_score.unique_onsets)),
        "time_shift_sec": shift,
        "messages_dropped_before_start": int(dropped),
        "polling_period": float(polling_period),
        "wall_time_sec": elapsed,
        "score_follower": type(accompanion.score_follower).__name__,
        "tempo_model": type(tempo_model).__name__,
    }
    return onsets, frames, alignment, accompaniment, stats


# ---------------------------------------------------------------------------
# Online against offline
# ---------------------------------------------------------------------------
def compare(knots, onsets, frames):
    """How far the online estimate is from the offline map.

    Position errors are in beats, positive when the online estimate is ahead
    of where the alignment says the player was. Tempo is compared as the
    ratio of the online beat period to the offline one at the same onset.
    """
    if len(knots) < 2 or len(onsets) == 0:
        return {}
    t_knots, b_knots = warp_table(knots)
    t0, t1 = t_knots[0], t_knots[-1]

    truth_at_onsets = np.interp(onsets["perf_onset_sec"], t_knots, b_knots)
    onset_error = onsets["score_onset_beat"] - truth_at_onsets

    ratio = onsets["beat_period"] / np.interp(
        onsets["score_onset_beat"], knots["score_onset_beat"], knots["beat_period"]
    )

    inside = (frames["time_sec"] >= t0) & (frames["time_sec"] <= t1) & ~frames["waiting"]
    truth_at_frames = np.interp(frames["time_sec"][inside], t_knots, b_knots)
    frame_error = frames["expected_position_beat"][inside] - truth_at_frames

    def summary(values):
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {}
        return {
            "median": float(np.median(values)),
            "abs_median": float(np.median(np.abs(values))),
            "abs_p95": float(np.percentile(np.abs(values), 95)),
            "max_abs": float(np.max(np.abs(values))),
        }

    return {
        "onset_position_error_beats": summary(onset_error),
        "onsets_off_by_more_than_2_beats": float(100.0 * np.mean(np.abs(onset_error) > 2)),
        "frame_position_error_beats": summary(frame_error),
        "tempo_ratio_online_over_offline": {
            "median": float(np.nanmedian(ratio)),
            "iqr": float(np.subtract(*np.nanpercentile(ratio, [75, 25]))),
            "p05": float(np.nanpercentile(ratio, 5)),
            "p95": float(np.nanpercentile(ratio, 95)),
        },
    }


# ---------------------------------------------------------------------------
# Driving it
# ---------------------------------------------------------------------------
def discover_takes(root, patterns):
    """``[(duo, take, {player: midi file})]`` under `root`, filtered."""
    takes = []
    for take_dir in sorted(glob.glob(os.path.join(root, "*", "*"))):
        if not os.path.isdir(take_dir):
            continue
        duo, take = os.path.basename(os.path.dirname(take_dir)), os.path.basename(take_dir)
        key = f"{duo}/{take}"
        if patterns and not any(fnmatch.fnmatch(key, p) for p in patterns):
            continue
        files = {
            os.path.basename(fn)[len("midi_"):-len(".mid")]: fn
            for fn in sorted(glob.glob(os.path.join(take_dir, "midi_p*.mid")))
        }
        if files:
            takes.append((duo, take, files))
    return takes


def piece_for_take(take, block_pieces):
    block = take.split("_", 1)[0]
    if block not in block_pieces:
        raise SystemExit(
            f"Take '{take}' starts with '{block}', which no --block-piece names "
            f"a piece for (have {sorted(block_pieces)})."
        )
    return block_pieces[block]


def process_take(args, pieces, matcher, duo, take, files, roles=None):
    """Run the chosen methods on one take and write its files.

    Returns the take's summary dict.
    """
    piece_name = piece_for_take(take, args.block_pieces)
    if piece_name not in pieces:
        pieces[piece_name] = Piece(piece_name)
    piece = pieces[piece_name]

    performances = {
        player: pt.load_performance_midi(fn)[0] for player, fn in files.items()
    }
    if roles is None:
        roles, margin = assign_roles(
            {player: part.note_array() for player, part in performances.items()},
            piece,
        )
    else:
        margin = None

    out_dir = os.path.join(args.out, duo, take)
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "duo": duo,
        "take": take,
        "piece": piece_name,
        "roles": roles,
        "role_margin": margin,
        "players": {},
    }

    for player, midi_fn in sorted(files.items()):
        role = roles[player]
        arrays = {"role": np.array(role), "midi_file": np.array(midi_fn)}
        info = {"role": role, "midi_file": midi_fn}

        if "offline" in args.methods:
            knots, alignment, raw_alignment, stats = offline_map(
                piece, role, performances[player], matcher, args.window
            )
            arrays["offline_knots"] = knots
            arrays["offline_alignment"] = alignment
            info["offline"] = stats
            if args.save_match:
                match_fn = os.path.join(out_dir, f"{player}_{role}.match")
                pt.save_match(
                    raw_alignment, performances[player], piece.parts[role],
                    out=match_fn, piece=piece_name, performer=f"{duo}_{player}",
                )
                info["offline"]["match_file"] = match_fn

        follow = "online" in args.methods and (
            role == "primo" or args.follow == "both"
        )
        if follow:
            accompanion = build_accompanion(args, piece, role)
            knots = arrays.get("offline_knots")
            first_note = (
                float(knots["perf_onset_sec"][0]) if knots is not None and len(knots) else None
            )
            onsets, frames, alignment, accompaniment, stats = online_map(
                accompanion, piece, role, midi_fn,
                lead_in=args.lead_in, first_note=first_note,
            )
            arrays["online_onsets"] = onsets
            arrays["online_frames"] = frames
            arrays["online_alignment"] = alignment
            arrays["online_accompaniment"] = accompaniment
            info["online"] = stats
            if "offline_knots" in arrays:
                info["online_vs_offline"] = compare(arrays["offline_knots"], onsets, frames)

        np.savez(os.path.join(out_dir, f"{player}.npz"), **arrays)
        summary["players"][player] = info

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def describe(summary):
    """One line per player, for the terminal."""
    lines = []
    for player, info in sorted(summary["players"].items()):
        parts = [f"  {player} {info['role']:<8s}"]
        off = info.get("offline")
        if off:
            parts.append(
                f"offline: {off['matched']}/{off['performed_notes']} notes matched, "
                f"{off.get('knots', 0)} knots"
            )
        on = info.get("online")
        if on:
            parts.append(
                f"online: {on['distinct_onsets']}/{on['score_onsets']} onsets, "
                f"{on['tempo_updates']} tempo updates"
            )
        cmp_ = info.get("online_vs_offline") or {}
        pos = cmp_.get("onset_position_error_beats") or {}
        ratio = cmp_.get("tempo_ratio_online_over_offline") or {}
        if pos:
            parts.append(
                f"err {pos['abs_median']:.2f}/{pos['abs_p95']:.2f} beats (p50/p95), "
                f"tempo ratio {ratio['median']:.3f}"
            )
        lines.append("  ".join(parts))
    return "\n".join(lines)


def parse_block_pieces(values):
    mapping = dict(DEFAULT_BLOCK_PIECES)
    for value in values or []:
        block, _, piece = value.partition("=")
        if not piece:
            raise SystemExit(f"--block-piece expects BLOCK=PIECE, got '{value}'")
        mapping[block] = piece
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Extract tempo curves from recorded duo performances.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_argument_group("input")
    source.add_argument("--dataset", help="root holding <duo>/<take>/midi_p*.mid")
    source.add_argument(
        "--takes", nargs="+", metavar="GLOB",
        help="only takes matching, e.g. 'D05/*' or '*/B1_*'",
    )
    source.add_argument("--midi", help="a single performance MIDI file instead")
    source.add_argument("--piece", help="with --midi: the piece it performs")
    source.add_argument(
        "--role", choices=ROLES, help="with --midi: the part it performs"
    )
    source.add_argument(
        "--block-piece", nargs="+", metavar="BLOCK=PIECE", dest="block_pieces",
        help="which piece a take block records. Default: "
        + " ".join(f"{k}={v}" for k, v in DEFAULT_BLOCK_PIECES.items()),
    )

    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--methods", nargs="+", choices=["online", "offline"],
        default=["online", "offline"],
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="redo takes that already have a summary.json")

    online = parser.add_argument_group("online (ACCompanion)")
    online.add_argument(
        "--config",
        help="config_files/<NAME>.yml to build the ACCompanion from. "
        "Default: the config named after the piece.",
    )
    online.add_argument(
        "--follower",
        help="build with this score follower instead of a config file, as "
        "bin/test_accompaniment.py does (e.g. PitchIOIHMM, outerhmm)",
    )
    online.add_argument("--follower-kwargs", metavar="JSON")
    online.add_argument("--polling-period", type=float)
    online.add_argument("--init-bpm", type=float)
    online.add_argument(
        "--follow", choices=["primo", "both"], default="primo",
        help="which player the ACCompanion follows. 'both' also follows the "
        "secondo as if it were the solo. Default: primo",
    )
    online.add_argument(
        "--lead-in", type=float, default=LEAD_IN_SEC,
        help=f"seconds between launching the ACCompanion and the first note. "
        f"Default: {LEAD_IN_SEC}",
    )

    offline = parser.add_argument_group("offline (parangonar)")
    offline.add_argument(
        "--matcher", choices=["dualdtw", "automatic"], default="dualdtw"
    )
    offline.add_argument(
        "--window", type=float, default=2.0,
        help="half-width, in beats, of the window the local tempo is the "
        "median over. Default: 2",
    )
    offline.add_argument(
        "--save-match", action="store_true",
        help="also write the alignment as a partitura .match file",
    )
    args = parser.parse_args()
    args.block_pieces = parse_block_pieces(args.block_pieces)
    # Progress is worth seeing as it happens when the output goes to a file.
    sys.stdout.reconfigure(line_buffering=True)

    if args.midi:
        if not (args.piece and args.role):
            parser.error("--midi needs --piece and --role")
        player = os.path.splitext(os.path.basename(args.midi))[0]
        takes = [("single", args.piece, {player: args.midi})]
        fixed_roles = {player: args.role}
        args.block_pieces = {args.piece: args.piece}
    elif args.dataset:
        takes = discover_takes(args.dataset, args.takes)
        fixed_roles = None
        if not takes:
            parser.error(f"no takes found under {args.dataset}")
    else:
        parser.error("give --dataset or --midi")

    matcher = make_matcher(args.matcher) if "offline" in args.methods else None
    pieces = {}
    index = []
    started = time.perf_counter()
    print(f"{len(takes)} take(s), methods: {', '.join(args.methods)}\n")
    for i, (duo, take, files) in enumerate(takes, 1):
        summary_fn = os.path.join(args.out, duo, take, "summary.json")
        if os.path.exists(summary_fn) and not args.overwrite:
            with open(summary_fn) as f:
                index.append(json.load(f))
            print(f"[{i}/{len(takes)}] {duo}/{take}: already done")
            continue
        print(f"[{i}/{len(takes)}] {duo}/{take}")
        try:
            summary = process_take(args, pieces, matcher, duo, take, files, roles=fixed_roles)
        except Exception as exc:  # noqa: BLE001 - one bad take must not stop the run
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            index.append({"duo": duo, "take": take, "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(describe(summary))
        index.append(summary)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n{len(index)} take(s) in {time.perf_counter() - started:.0f}s -> {args.out}/index.json")


if __name__ == "__main__":
    main()
