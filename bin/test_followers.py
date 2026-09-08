#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline comparison of the score followers the ACCompanion can run.

Feeds a performance through a follower exactly the way the ACCompanion's main
loop does -- MIDI frames into the input pipeline, its output into the score
follower -- but without MIDI hardware, audio, or the accompaniment. Use it to
see which trackers work on a piece and how closely they follow.

Examples
--------
List every available follower::

    python bin/test_followers.py --list

Run all of them on a sample piece, against a performance rendered from the
solo score itself (so the true score position is known at every moment)::

    python bin/test_followers.py --piece bach_menuett

Run a few, on a real MIDI performance::

    python bin/test_followers.py --piece bach_menuett \
        --followers hmm arzt PitchIOIHMM --midi-fn my_performance.mid
"""
import argparse
import os
import sys
import time
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mido
import numpy as np
import partitura as pt

warnings.filterwarnings("ignore", module="partitura")

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


def messages_from_score(solo_fn, bpm):
    """A performance of the solo score at a constant tempo."""
    part = pt.load_score(solo_fn)[0]
    ppart = pt.utils.music.performance_from_part(part, bpm=bpm)
    return _messages_from_note_array(ppart.note_array())


def messages_from_midi(midi_fn):
    """A performance read from a MIDI file."""
    performance = pt.load_performance(midi_fn)
    return _messages_from_note_array(performance.note_array())


def _messages_from_note_array(note_array):
    messages, times = [], []
    for note in note_array:
        pitch, velocity = int(note["pitch"]), int(note["velocity"])
        messages.append(mido.Message("note_on", note=pitch, velocity=velocity))
        times.append(float(note["onset_sec"]))
        messages.append(mido.Message("note_off", note=pitch, velocity=velocity))
        times.append(float(note["onset_sec"] + note["duration_sec"]))
    order = np.argsort(times, kind="stable")
    return [messages[i] for i in order], [times[i] for i in order]


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


def build_accompanion(follower, solo_fn, acc_fn, polling_period, init_bpm):
    """An ACCompanion with its scores and score follower set up, nothing else."""
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
            "score_follower_kwargs": {},
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
            "score_follower_kwargs": {},
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
    )
    accompanion.setup_scores()
    accompanion.setup_score_follower()
    return accompanion


def run_follower(accompanion, frames):
    """Drive the follower over `frames`, the way `ACCompanion.run` does."""
    positions, latencies = [], []
    for frame in frames:
        output = accompanion.input_pipeline(frame)
        accompanion.check_empty_frames(output)
        start = time.perf_counter()
        position = accompanion.score_follower(output)
        latencies.append(time.perf_counter() - start)
        if position is not None:
            positions.append((frame[1], float(position)))
    return positions, np.array(latencies)


def evaluate(accompanion, positions, latencies, beat_period, has_ground_truth):
    onsets = accompanion.solo_score.unique_onsets
    result = {
        "emitted": len(positions),
        "onsets": len(onsets),
        "latency_ms": 1000 * latencies.mean() if len(latencies) else float("nan"),
        "max_latency_ms": 1000 * latencies.max() if len(latencies) else float("nan"),
    }
    if not positions:
        result["note"] = "no positions reported"
        return result

    reported = np.array([p for _, p in positions])
    result["monotonic"] = bool(np.all(np.diff(reported) >= -1e-9))
    result["final"] = reported[-1]
    result["last_onset"] = float(onsets.max())

    if has_ground_truth:
        # Constant tempo, so the true position is linear in performance time.
        true = np.array([t / beat_period + onsets.min() for t, _ in positions])
        error = np.abs(reported - true)
        result["mean_err"] = float(error.mean())
        result["max_err"] = float(error.max())
    return result


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
        "--followers",
        nargs="+",
        help="followers to run. Default: all of them.",
    )
    parser.add_argument(
        "--midi-fn",
        help="MIDI performance to follow. Without it, a performance is "
        "rendered from the solo score at --bpm, which is what makes the "
        "tracking error columns meaningful.",
    )
    parser.add_argument("--bpm", type=float, default=110.0, help="tempo. Default: 110")
    parser.add_argument(
        "--polling-period",
        type=float,
        help="override the polling period. Default: whatever each follower asks for.",
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
        parser.error(
            f"unknown follower(s) {unknown}. Available: {all_followers}"
        )

    if args.midi_fn:
        messages, times = messages_from_midi(args.midi_fn)
        has_ground_truth = False
        source = os.path.basename(args.midi_fn)
    else:
        messages, times = messages_from_score(solo_fn, args.bpm)
        has_ground_truth = True
        source = f"solo score rendered at {args.bpm:g} bpm"

    print(f"piece:       {os.path.dirname(solo_fn)}")
    print(f"performance: {source} ({len(messages)} messages, "
          f"{max(times):.1f}s)")
    print()

    header = (
        f"{'follower':16s} {'emitted':>8s} {'onsets':>7s} {'mono':>5s} "
        f"{'mean err':>9s} {'max err':>8s} {'latency':>9s} {'max lat':>8s}"
    )
    print(header)
    print("-" * len(header))

    results = {}
    for follower in followers:
        polling_period = args.polling_period
        if polling_period is None:
            wanted = (
                preferred_polling_period(follower)
                if follower not in ACCOMPANION_FOLLOWERS
                else DEFAULT_POLLING_PERIOD
            )
            # A null polling period means the follower wants one message at a
            # time; the ACCompanion still needs a number for its own loop.
            polling_period = EVENT_POLLING_PERIOD if wanted is None else wanted

        try:
            accompanion = build_accompanion(
                follower, solo_fn, acc_fn, polling_period, args.bpm
            )
            if getattr(accompanion, "event_based_input", False):
                frames = event_frames(messages, times)
            else:
                frames = framed(messages, times, polling_period)
            positions, latencies = run_follower(accompanion, frames)
            result = evaluate(
                accompanion, positions, latencies, 60 / args.bpm, has_ground_truth
            )
        except Exception as exc:  # noqa: BLE001 - a failing follower is a result
            if args.traceback:
                traceback.print_exc()
            print(f"{follower:16s} FAILED: {type(exc).__name__}: {exc}")
            results[follower] = {"error": exc}
            continue

        results[follower] = result
        mono = "yes" if result.get("monotonic") else "no"
        mean_err = result.get("mean_err")
        max_err = result.get("max_err")
        print(
            f"{follower:16s} {result['emitted']:8d} {result['onsets']:7d} "
            f"{mono:>5s} "
            f"{(f'{mean_err:.3f}' if mean_err is not None else '-'):>9s} "
            f"{(f'{max_err:.3f}' if max_err is not None else '-'):>8s} "
            f"{result['latency_ms']:8.3f}ms {result['max_latency_ms']:7.2f}ms"
        )

    print()
    print("emitted   score positions reported to the accompanist")
    print("onsets    unique score onsets in the solo part")
    print("mono      whether the reported positions never go backwards")
    print("mean/max err   tracking error in beats, against the known true position")
    print("latency   time spent inside the score follower per input frame")

    failed = [name for name, r in results.items() if "error" in r]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
