#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline comparison of the score followers the ACCompanion can run.

Feeds a performance through a follower exactly the way the ACCompanion's main
loop does -- MIDI frames into the input pipeline, its output into the score
follower -- but without MIDI hardware, audio, or the accompaniment.

The performance is synthesised from the solo score, so the true score position
is known at every instant. It can be synthesised *badly* on purpose: the
scenarios below bend the tempo, smear the timing and corrupt the notes, which
is what separates a follower that stays locked on a clean rendering from one
that survives a real player.

Examples
--------
List every available follower::

    python bin/test_followers.py --list

Rank the followers on a clean rendering (fast, the easiest possible input)::

    python bin/test_followers.py --piece bach_menuett

Rank them on more realistic playing, three random seeds per scenario::

    python bin/test_followers.py --piece bach_menuett --scenarios all --seeds 3

A few followers, on a real MIDI recording (no ground truth, so only the
coverage and latency columns are meaningful)::

    python bin/test_followers.py --piece bach_menuett --followers hmm arzt \
        --midi-fn my_performance.mid
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mido
import numpy as np
import partitura as pt

warnings.filterwarnings("ignore", module="partitura")

from accompanion.accompanist.fermata import (
    fermata_onsets_from_part,
    free_sections_from_part,
    waiting_points,
)
from accompanion.score_follower.matchmaker_methods import (
    available_methods,
    preferred_polling_period,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIECE_DIRS = [
    os.path.join(REPO_ROOT, "sample_pieces"),
    os.path.join(REPO_ROOT, "accompanion_pieces", "simple_pieces"),
    os.path.join(REPO_ROOT, "accompanion_pieces", "complex_pieces"),
]

#: The ACCompanion's own followers, and the class that runs each.
ACCOMPANION_FOLLOWERS = {
    "PitchIOIHMM": "hmm",
    "PitchIOIKHMM": "hmm",
    "OnlineTimeWarping": "oltw",
}

DEFAULT_POLLING_PERIOD = 0.01
# What the ACCompanion uses for a follower that wants one message per frame.
EVENT_POLLING_PERIOD = 0.001

# A follower is counted as lost while it is this far from the true position.
LOST_THRESHOLD_BEATS = 2.0
# How long every follower is given to react to an onset before its reported
# position is read. Followers differ in how often they speak -- an HMM only at
# onsets, an OLTW on every frame, holding its position in between -- so the
# position has to be sampled at the same moment relative to the note for the
# comparison to mean anything.
REACTION_ALLOWANCE_S = 0.05


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """How the solo part is played.

    Every field is something a real player does and a constant-tempo rendering
    of the score does not.
    """

    name: str
    #: Peak tempo deviation as a fraction, as a sine over `rubato_period` beats.
    rubato: float = 0.0
    rubato_period: float = 8.0
    #: Fractional slowdown reached at the very end of the piece.
    final_rit: float = 0.0
    #: Standard deviation of per-note onset noise, in milliseconds.
    jitter_ms: float = 0.0
    #: Notes of a chord are rolled over this long, in milliseconds.
    chord_spread_ms: float = 0.0
    #: Fraction of notes played at the wrong pitch.
    wrong: float = 0.0
    #: Fraction of notes not played at all.
    missed: float = 0.0
    #: Fraction of notes followed by a spurious extra note.
    extra: float = 0.0
    #: Seconds a fermata is held for, beyond the note's notated length. Drawn
    #: per fermata around this value, because that is the whole point of a
    #: fermata: how long it lasts is up to the player, and nothing in the
    #: score or the tempo predicts it.
    fermata_s: float = 0.0
    #: The same, at every onset inside a section marked to be played freely.
    free_s: float = 0.0

    @property
    def is_random(self) -> bool:
        """Whether repeated runs differ, i.e. whether several seeds are useful."""
        return any(
            [
                self.jitter_ms,
                self.wrong,
                self.missed,
                self.extra,
                self.chord_spread_ms,
                self.fermata_s,
                self.free_s,
            ]
        )


SCENARIOS = {
    s.name: s
    for s in [
        Scenario("clean"),
        # A player shaping the phrase, and slowing into the final bars.
        Scenario("rubato", rubato=0.15, rubato_period=8.0, final_rit=0.35),
        # Uneven hands: notes land early or late, chords are rolled.
        Scenario("jitter", jitter_ms=30.0, chord_spread_ms=25.0),
        # Wrong notes, dropped notes, and the odd stray one.
        Scenario("errors", wrong=0.05, missed=0.05, extra=0.03),
        # All of it at once, each a bit milder: roughly an amateur run-through.
        Scenario(
            "human",
            rubato=0.10,
            rubato_period=8.0,
            final_rit=0.25,
            jitter_ms=20.0,
            chord_spread_ms=20.0,
            wrong=0.02,
            missed=0.02,
            extra=0.01,
        ),
        # Otherwise metronomic, but the fermatas and any free section are
        # taken in the player's own time. Nothing here is a following error:
        # the score itself says the beat stops at these points, and an
        # accompaniment that keeps counting through them is simply wrong.
        Scenario("fermata", fermata_s=2.5, free_s=1.5),
        # The same holds, played by the same amateur as 'human'.
        Scenario(
            "freehand",
            rubato=0.10,
            rubato_period=8.0,
            final_rit=0.25,
            jitter_ms=20.0,
            chord_spread_ms=20.0,
            wrong=0.02,
            missed=0.02,
            extra=0.01,
            fermata_s=2.5,
            free_s=1.5,
        ),
    ]
}
DEFAULT_SCENARIOS = ["clean"]


def find_piece(name):
    """Locate primo.musicxml / secondo.musicxml for a named piece."""
    for base in PIECE_DIRS:
        directory = os.path.join(base, name)
        solo = os.path.join(directory, "primo.musicxml")
        acc = os.path.join(directory, "secondo.musicxml")
        if os.path.exists(solo) and os.path.exists(acc):
            return solo, acc
    raise SystemExit(
        f"Could not find piece '{name}'. Looked for primo.musicxml and "
        f"secondo.musicxml in:\n  " + "\n  ".join(PIECE_DIRS)
    )


# ---------------------------------------------------------------------------
# Synthesising a performance
# ---------------------------------------------------------------------------
#: How wide the step a hold makes in the beat -> time map is, in beats. Small
#: enough to read as an instant, large enough to keep the map invertible.
HOLD_EPSILON = 1e-6


def score_holds(solo_fn, scenario, seed):
    """Where the synthesised soloist stops, and for how long.

    Fermatas and free sections are read from the score exactly as the
    ACCompanion reads them, and each stop is drawn separately: how long a
    fermata lasts is the player's own decision, and two players -- or the same
    player twice -- will not agree.

    Returns
    -------
    list of (float, float)
        ``(score beat, seconds held there)``.
    """
    if not (scenario.fermata_s or scenario.free_s):
        return []

    part = pt.load_score(solo_fn)[0]
    onsets = np.unique(part.note_array()["onset_beat"])
    rng = np.random.default_rng(seed + 977)

    holds = []
    if scenario.fermata_s:
        for beat in waiting_points(onsets, fermata_onsets_from_part(part)):
            holds.append((float(beat), scenario.fermata_s * float(rng.uniform(0.5, 1.5))))
    if scenario.free_s:
        for beat in waiting_points(
            onsets, free_sections=free_sections_from_part(part)
        ):
            holds.append((float(beat), scenario.free_s * float(rng.uniform(0.2, 1.8))))

    # A beat that is both a fermata and inside a free section is held once.
    longest = {}
    for beat, seconds in holds:
        longest[beat] = max(longest.get(beat, 0.0), seconds)
    return sorted(longest.items())


def beat_to_time_map(beats_min, beats_max, bpm, scenario, n=4001, holds=()):
    """Maps between score beats and performance seconds for a scenario.

    The tempo curve is integrated over the beat axis, so a beat position maps
    to the time it is actually played at, and the inverse gives the true score
    position at any instant -- the ground truth the followers are scored on.

    A hold is a step in that map: at a fermata the clock runs on while the
    score position does not, which is exactly what makes fermatas hard for an
    accompaniment that dead-reckons.

    Parameters
    ----------
    holds : iterable of (float, float)
        ``(score beat, seconds held there)``, from `score_holds`.

    Returns
    -------
    to_time : callable, beats -> seconds
    to_beat : callable, seconds -> beats
    """
    holds = list(holds)
    beats = np.linspace(beats_min, beats_max, n)
    if holds:
        # Sample on both sides of every hold, so the step in the map is a step
        # rather than a ramp across a whole linspace interval.
        beats = np.unique(
            np.concatenate(
                [beats] + [[beat, beat + HOLD_EPSILON] for beat, _ in holds]
            )
        )
    n = len(beats)
    base = 60.0 / bpm
    period = np.full(n, base)

    if scenario.rubato:
        period = period * (
            1.0 + scenario.rubato * np.sin(2 * np.pi * beats / scenario.rubato_period)
        )
    if scenario.final_rit:
        # Ramp over the last eighth of the piece.
        span = max((beats_max - beats_min) / 8.0, 1e-9)
        ramp = np.clip((beats - (beats_max - span)) / span, 0.0, 1.0)
        period = period * (1.0 + scenario.final_rit * ramp)

    # times[i] = integral of the beat period up to beats[i]
    times = np.concatenate([[0.0], np.cumsum(np.diff(beats) * period[:-1])])

    for beat, seconds in holds:
        # Everything after the hold happens that much later; the beat the
        # hold sits on is still reached on time.
        times = times + seconds * (beats > beat)

    def to_time(b):
        return np.interp(b, beats, times)

    def to_beat(t):
        return np.interp(t, times, beats)

    return to_time, to_beat


def beat_to_time_map_for(solo_fn, bpm, scenario, seed=0):
    """The `(beats -> seconds, seconds -> beats)` maps `render` plays through.

    Pass the same `seed` as `render`, or the holds will not line up.
    """
    note_array = pt.load_score(solo_fn)[0].note_array()
    return beat_to_time_map(
        float(note_array["onset_beat"].min()),
        float((note_array["onset_beat"] + note_array["duration_beat"]).max()),
        bpm,
        scenario,
        holds=score_holds(solo_fn, scenario, seed),
    )


def render(solo_fn, bpm, scenario, seed):
    """Synthesise a performance of the solo part.

    Returns
    -------
    messages : list of mido.Message
    times : list of float, seconds
    to_beat : callable, seconds -> true score beat
    onsets : np.ndarray, the unique score onsets, in beats
    onset_times : np.ndarray, when each of them is played, in seconds
    """
    rng = np.random.default_rng(seed)
    note_array = pt.load_score(solo_fn)[0].note_array()
    onsets = np.unique(note_array["onset_beat"])

    to_time, to_beat = beat_to_time_map(
        float(note_array["onset_beat"].min()),
        float((note_array["onset_beat"] + note_array["duration_beat"]).max()),
        bpm,
        scenario,
        holds=score_holds(solo_fn, scenario, seed),
    )

    messages, times = [], []
    # Position within its chord, so a rolled chord spreads in pitch order.
    order_in_chord = {}
    for onset in onsets:
        pitches = np.sort(note_array["pitch"][note_array["onset_beat"] == onset])
        for i, p in enumerate(pitches):
            order_in_chord[(float(onset), int(p))] = i

    for note in note_array:
        if scenario.missed and rng.random() < scenario.missed:
            continue

        pitch = int(note["pitch"])
        if scenario.wrong and rng.random() < scenario.wrong:
            pitch = int(np.clip(pitch + rng.choice([-2, -1, 1, 2]), 21, 108))

        on = float(to_time(note["onset_beat"]))
        off = float(to_time(note["onset_beat"] + max(note["duration_beat"], 1e-3)))

        if scenario.chord_spread_ms:
            k = order_in_chord.get((float(note["onset_beat"]), int(note["pitch"])), 0)
            on += k * scenario.chord_spread_ms / 1000.0
        if scenario.jitter_ms:
            on += float(rng.normal(0.0, scenario.jitter_ms / 1000.0))
        off = max(off, on + 0.02)

        velocity = int(note["velocity"]) if "velocity" in note_array.dtype.names else 64
        messages.append(mido.Message("note_on", note=pitch, velocity=velocity))
        times.append(on)
        messages.append(mido.Message("note_off", note=pitch, velocity=velocity))
        times.append(off)

        if scenario.extra and rng.random() < scenario.extra:
            stray = int(np.clip(pitch + rng.choice([-4, -3, 3, 4]), 21, 108))
            t0 = on + float(rng.uniform(0.02, 0.15))
            messages.append(mido.Message("note_on", note=stray, velocity=velocity))
            times.append(t0)
            messages.append(mido.Message("note_off", note=stray, velocity=velocity))
            times.append(t0 + 0.08)

    order = np.argsort(times, kind="stable")
    messages = [messages[i] for i in order]
    times = [float(times[i]) for i in order]
    return messages, times, to_beat, onsets, np.asarray(to_time(onsets), dtype=float)


def messages_from_midi(midi_fn):
    """A performance read from a MIDI file. No ground truth."""
    note_array = pt.load_performance(midi_fn).note_array()
    messages, times = [], []
    for note in note_array:
        pitch, velocity = int(note["pitch"]), int(note["velocity"])
        messages.append(mido.Message("note_on", note=pitch, velocity=velocity))
        times.append(float(note["onset_sec"]))
        messages.append(mido.Message("note_off", note=pitch, velocity=velocity))
        times.append(float(note["onset_sec"] + note["duration_sec"]))
    order = np.argsort(times, kind="stable")
    return [messages[i] for i in order], [float(times[i]) for i in order]


# ---------------------------------------------------------------------------
# Framing, as the MIDI input thread does it
# ---------------------------------------------------------------------------
def framed(messages, times, polling_period):
    """Frames of `polling_period`, as `FramedMidiInputThread` builds them."""
    n_frames = int(np.ceil(max(times) / polling_period)) + 1
    frames, cursor = [], 0
    for i in range(n_frames):
        end = (i + 1) * polling_period
        frame = []
        while cursor < len(times) and times[cursor] < end:
            frame.append((messages[cursor], times[cursor]))
            cursor += 1
        frames.append((frame, i * polling_period + 0.5 * polling_period))
    return frames


def event_frames(messages, times):
    """One frame per MIDI message, as `MidiInputThread` builds them."""
    return [([(msg, t)], t) for msg, t in zip(messages, times)]


# ---------------------------------------------------------------------------
# Running a follower
# ---------------------------------------------------------------------------
def build_accompanion(
    follower, solo_fn, acc_fn, polling_period, init_bpm, follower_kwargs=None,
    setup=True, fermata_kwargs=None,
):
    """An ACCompanion with its scores and score follower set up, nothing else.

    Pass ``setup=False`` to get it unconfigured, for a caller that wants to run
    `setup_following` instead and drive the whole accompaniment chain.
    `fermata_kwargs` reaches the accompaniment's waiting at fermatas, which
    only `setup_following` builds.
    """
    router_kwargs = dict(
        solo_input_to_accompaniment_port_name=None,
        acc_output_to_sound_port_name=None,
        MIDIPlayer_to_sound_port_name=None,
        MIDIPlayer_to_accompaniment_port_name=None,
    )
    kind = ACCOMPANION_FOLLOWERS.get(follower, "matchmaker")

    if kind == "hmm":
        from accompanion.hmm_accompanion import HMMACCompanion as cls

        score_follower_kwargs = {
            "score_follower": follower,
            "score_follower_kwargs": dict(follower_kwargs or {}),
            "input_processor": {
                "processor": "PitchIOIProcessor",
                "processor_kwargs": {},
            },
        }
    elif kind == "oltw":
        from accompanion.oltw_accompanion import OLTWACCompanion as cls

        score_follower_kwargs = {
            "score_follower": follower,
            "window_size": 100,
            "step_size": 10,
            "input_processor": {
                "processor": "PianoRollProcessor",
                "processor_kwargs": {"piano_range": True},
            },
        }
    else:
        from accompanion.matchmaker_accompanion import MatchmakerACCompanion as cls

        score_follower_kwargs = {
            "score_follower": follower,
            "score_follower_kwargs": dict(follower_kwargs or {}),
        }

    accompanion = cls(
        solo_fn=solo_fn,
        acc_fn=acc_fn,
        midi_router_kwargs=router_kwargs,
        score_follower_kwargs=score_follower_kwargs,
        tempo_model_kwargs={"tempo_model": "LSM"},
        polling_period=polling_period,
        init_bpm=init_bpm,
        test=True,
        fermata_kwargs=fermata_kwargs,
    )
    if setup:
        accompanion.setup_scores()
        accompanion.setup_score_follower()
    return accompanion


def run_follower(accompanion, frames):
    """Drive the follower over `frames`, the way `ACCompanion.run` does."""
    positions, latencies = [], []
    for frame in frames:
        output = accompanion.input_pipeline(frame)
        # base.py uses check_empty_frames only to count idle loops; the score
        # follower is called on every frame.
        accompanion.check_empty_frames(output)
        start = time.perf_counter()
        position = accompanion.score_follower(output)
        latencies.append(time.perf_counter() - start)
        if position is not None:
            positions.append((frame[1], float(position)))
    return positions, np.array(latencies)


def polling_period_for(follower, override=None):
    """The polling period a follower is fed at."""
    if override is not None:
        return override
    wanted = (
        DEFAULT_POLLING_PERIOD
        if follower in ACCOMPANION_FOLLOWERS
        else preferred_polling_period(follower)
    )
    # A null polling period means one message at a time; the ACCompanion still
    # needs a number for its own loop.
    return EVENT_POLLING_PERIOD if wanted is None else wanted


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_run(positions, latencies, to_beat, onsets, onset_times):
    """Turn one run into the numbers the table reports."""
    result = {
        "emitted": len(positions),
        "cpu_ms": 1000 * latencies.mean() if len(latencies) else float("nan"),
        "cpu_max_ms": 1000 * latencies.max() if len(latencies) else float("nan"),
    }
    if not positions:
        return result

    t = np.array([p[0] for p in positions])
    reported = np.array([p[1] for p in positions])

    if to_beat is not None:
        # Error is read a fixed moment after each onset is played, so that
        # every follower gets the same chance to react. Averaging over emitted
        # frames instead would charge a follower that holds its position
        # between onsets for the hold rather than for being wrong -- half a
        # median IOI of it, enough to invert the ranking.
        idx = np.searchsorted(t, onset_times + REACTION_ALLOWANCE_S, side="right") - 1
        seen = idx >= 0
        if seen.any():
            err = np.abs(reported[idx[seen]] - onsets[seen])
            result["err50"] = float(np.percentile(err, 50))
            result["err95"] = float(np.percentile(err, 95))
            result["lost"] = float(100.0 * np.mean(err > LOST_THRESHOLD_BEATS))

        # How long after a score onset is actually played does the follower
        # first report having reached it? This is what the accompanist waits on.
        run_max = np.maximum.accumulate(reported)
        reach_idx = np.searchsorted(run_max, onsets, side="left")
        reached = reach_idx < len(t)
        result["missed"] = float(100.0 * np.mean(~reached))
        if reached.any():
            lat = t[reach_idx[reached]] - onset_times[reached]
            result["lat50"] = float(1000 * np.percentile(lat, 50))
            result["lat95"] = float(1000 * np.percentile(lat, 95))

    result["back"] = int(np.sum(np.diff(reported) < -1e-9))
    return result


def aggregate(runs):
    """Mean of each metric over seeds."""
    out = {}
    for key in set().union(*(r.keys() for r in runs)):
        vals = [r[key] for r in runs if key in r]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


HEADER = (
    f"{'follower':<17s}{'scenario':<9s}{'err50':>7s}{'err95':>8s}"
    f"{'lat50':>8s}{'lat95':>8s}{'miss%':>7s}{'lost%':>7s}{'back':>6s}{'cpu':>9s}"
)
LEGEND = """
err50/err95  tracking error in beats, read 50 ms after each onset is played
             so every follower gets the same chance to react
lat50/lat95  how long after a score onset is played the follower reports
             reaching it, in ms -- what the accompanist waits on
miss%        score onsets the follower never reached
lost%        share of onsets the follower was more than 2 beats away from
back         times the reported position jumped backwards
cpu          mean time inside the follower per input frame"""


def format_row(follower, scenario, r):
    def num(key, fmt, scale=1.0):
        v = r.get(key)
        return format(v * scale, fmt) if v is not None and np.isfinite(v) else "-"

    return (
        f"{follower:<17s}{scenario:<9s}"
        f"{num('err50', '7.3f')}{num('err95', '8.3f')}"
        f"{num('lat50', '8.0f')}{num('lat95', '8.0f')}"
        f"{num('missed', '7.1f')}{num('lost', '7.1f')}"
        f"{int(round(r.get('back', 0))):6d}{num('cpu_ms', '7.2f')}ms"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare the ACCompanion's score followers offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", action="store_true", help="list the available followers and exit"
    )
    parser.add_argument("--piece", help="piece name, e.g. bach_menuett")
    parser.add_argument("--solo", help="solo score file (overrides --piece)")
    parser.add_argument("--acc", help="accompaniment score file (overrides --piece)")
    parser.add_argument(
        "--followers", nargs="+", help="followers to run. Default: all of them."
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=DEFAULT_SCENARIOS,
        help="how the solo part is played: "
        + ", ".join(SCENARIOS)
        + ", or 'all'. Default: clean",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="random seeds per scenario, averaged. Ignored for 'clean', which "
        "is deterministic. Default: 3",
    )
    parser.add_argument(
        "--midi-fn",
        help="follow a real MIDI recording instead of a synthesised "
        "performance. There is no ground truth then, so only the coverage and "
        "cpu columns are filled in.",
    )
    parser.add_argument("--bpm", type=float, default=110.0, help="tempo. Default: 110")
    parser.add_argument(
        "--polling-period",
        type=float,
        help="override the polling period. Default: whatever each follower asks for.",
    )
    parser.add_argument(
        "--follower-kwargs",
        metavar="JSON",
        help="method configuration for a matchmaker follower, as JSON. For "
        'example \'{"members": [{"method": "pthmm"}, {"method": "outerhmm"}]}\' '
        "to choose the ensemble's members.",
    )
    parser.add_argument(
        "--traceback", action="store_true", help="print tracebacks for failures"
    )
    args = parser.parse_args()

    all_followers = list(ACCOMPANION_FOLLOWERS) + available_methods()

    if args.list:
        print("ACCompanion's own score followers:")
        for name, kind in ACCOMPANION_FOLLOWERS.items():
            print(f"  {name:16s} ({kind} accompanion)")
        print("\nMatchmaker score followers:")
        for name in available_methods():
            pp = preferred_polling_period(name)
            unit = "event based" if pp is None else f"polling period {pp}s"
            print(f"  {name:16s} ({unit})")
        print("\nScenarios:")
        for name, sc in SCENARIOS.items():
            bits = []
            if sc.rubato:
                bits.append(f"rubato +-{sc.rubato:.0%}")
            if sc.final_rit:
                bits.append(f"final rit {sc.final_rit:.0%}")
            if sc.jitter_ms:
                bits.append(f"jitter {sc.jitter_ms:.0f}ms")
            if sc.chord_spread_ms:
                bits.append(f"rolled chords {sc.chord_spread_ms:.0f}ms")
            if sc.wrong:
                bits.append(f"{sc.wrong:.0%} wrong")
            if sc.missed:
                bits.append(f"{sc.missed:.0%} missed")
            if sc.extra:
                bits.append(f"{sc.extra:.0%} extra")
            if sc.fermata_s:
                bits.append(f"fermatas held ~{sc.fermata_s:.1f}s")
            if sc.free_s:
                bits.append(f"free sections ~{sc.free_s:.1f}s per note")
            print(f"  {name:16s} {', '.join(bits) if bits else 'exactly as written'}")
        return 0

    if args.solo and args.acc:
        solo_fn, acc_fn = args.solo, args.acc
    elif args.piece:
        solo_fn, acc_fn = find_piece(args.piece)
    else:
        parser.error("give either --piece or both --solo and --acc")

    followers = args.followers or all_followers
    unknown = [f for f in followers if f not in all_followers]
    if unknown:
        parser.error(f"unknown follower(s) {unknown}. Available: {all_followers}")

    if "all" in args.scenarios:
        scenario_names = list(SCENARIOS)
    else:
        unknown = [s for s in args.scenarios if s not in SCENARIOS]
        if unknown:
            parser.error(f"unknown scenario(s) {unknown}. Available: {list(SCENARIOS)}")
        scenario_names = args.scenarios

    follower_kwargs = json.loads(args.follower_kwargs) if args.follower_kwargs else None
    if follower_kwargs is not None and not isinstance(follower_kwargs, dict):
        parser.error("--follower-kwargs must be a JSON object")

    print(f"piece:  {os.path.dirname(solo_fn)}")
    if args.midi_fn:
        print(f"input:  {os.path.basename(args.midi_fn)} (no ground truth)")
        scenario_names = ["recording"]
    else:
        print(f"input:  solo score played at {args.bpm:g} bpm")
        print(f"scenarios: {', '.join(scenario_names)}   seeds: {args.seeds}")
    print()
    print(HEADER)
    print("-" * len(HEADER))

    failures = []
    for follower in followers:
        polling_period = polling_period_for(follower, args.polling_period)

        for scenario_name in scenario_names:
            scenario = SCENARIOS.get(scenario_name)
            seeds = [0] if scenario is None or not scenario.is_random else range(
                args.seeds
            )
            runs = []
            try:
                for seed in seeds:
                    if args.midi_fn:
                        messages, times = messages_from_midi(args.midi_fn)
                        to_beat, onsets, onset_times = None, None, None
                    else:
                        messages, times, to_beat, onsets, onset_times = render(
                            solo_fn, args.bpm, scenario, seed
                        )
                    accompanion = build_accompanion(
                        follower, solo_fn, acc_fn, polling_period, args.bpm,
                        follower_kwargs=follower_kwargs,
                    )
                    if getattr(accompanion, "event_based_input", False):
                        frames = event_frames(messages, times)
                    else:
                        frames = framed(messages, times, polling_period)
                    positions, latencies = run_follower(accompanion, frames)
                    runs.append(
                        score_run(positions, latencies, to_beat, onsets, onset_times)
                    )
            except Exception as exc:  # noqa: BLE001 - a failing follower is a result
                if args.traceback:
                    traceback.print_exc()
                print(f"{follower:<17s}{scenario_name:<9s}FAILED: "
                      f"{type(exc).__name__}: {exc}")
                failures.append((follower, scenario_name))
                continue

            print(format_row(follower, scenario_name, aggregate(runs)), flush=True)

    print(LEGEND)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
