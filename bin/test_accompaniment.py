#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline measurement of what the ACCompanion actually plays.

`test_followers.py` scores the score follower alone. This scores the whole
chain -- follower, onset tracker, tempo model, performance codec, sequencer --
by answering the question a listener would ask: *was the accompaniment on
time?*

It drives `ACCompanion.follow_step`, the same method the live loop calls, over
a synthesised solo performance, and simulates the sequencer: an accompaniment
note is played at the first instant the clock reaches the onset time the
accompanist has assigned it, exactly as `ScoreSequencer` does. Because the solo
performance is synthesised through a known tempo curve, the time each
accompaniment onset *should* have sounded is known too, and the difference
between the two is the number that matters.

Examples
--------
    python bin/test_accompaniment.py --piece badinerie
    python bin/test_accompaniment.py --piece badinerie --scenarios all --seeds 3
    python bin/test_accompaniment.py --piece badinerie \
        --followers PitchIOIHMM outerhmm ensemble
"""
import argparse
import contextlib
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import test_followers as H
from accompanion.base import FollowingState
from accompanion.score_follower.onset_tracker import OnsetTracker


def build(follower, solo_fn, acc_fn, polling_period, init_bpm, follower_kwargs=None,
          fermata_kwargs=None):
    """An ACCompanion with the whole following chain set up, minus the hardware."""
    accompanion = H.build_accompanion(
        follower, solo_fn, acc_fn, polling_period, init_bpm,
        follower_kwargs=follower_kwargs, setup=False,
        fermata_kwargs=fermata_kwargs,
    )
    accompanion.setup_following()
    return accompanion


class SimulatedSequencer(object):
    """`ScoreSequencer` without the MIDI port or the wall clock.

    Plays an accompaniment note the first time the clock reaches the onset the
    accompanist has assigned it, and records when that happened. The assigned
    onset keeps being revised until the note is played, which is why this has
    to be stepped alongside the loop rather than read off at the end.
    """

    def __init__(self, acc_score):
        self.notes = sorted(acc_score.notes, key=lambda n: n.onset)
        #: ``(score onset in beats, time played in seconds)`` per note.
        self.played = []

    def step(self, clock):
        for note in self.notes:
            if note.already_performed:
                continue
            p_onset = note.p_onset
            if p_onset is not None and clock >= p_onset:
                note.already_performed = True
                self.played.append((float(note.onset), float(clock)))


@contextlib.contextmanager
def quiet():
    """Swallow the ACCompanion's own progress printing.

    `follow_step` and the note tracker narrate every onset to stdout, which is
    useful when playing and unreadable when running hundreds of performances.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def run_once(accompanion, frames, to_time):
    """Follow one performance and return the accompaniment onsets played.

    Returns
    -------
    played : dict
        ``{accompaniment score onset: earliest time it sounded}``.
    steps : int
        How many accompaniment steps the accompanist took.
    """
    onset_tracker = OnsetTracker(accompanion.solo_score.unique_onsets)
    state = FollowingState(expected_position=accompanion.first_score_onset)
    sequencer = SimulatedSequencer(accompanion.acc_score)

    # The accompaniment leads if it starts before the solo, exactly as `run`
    # decides it.
    state.solo_starts = accompanion.acc_score.min_onset >= accompanion.solo_score.min_onset

    with quiet():
        for frame, frame_time in frames:
            accompanion.follow_step(
                input_midi_messages=frame,
                output=accompanion.input_pipeline((frame, frame_time)),
                solo_p_onset=frame_time,
                onset_tracker=onset_tracker,
                state=state,
            )
            sequencer.step(frame_time)

    played = {}
    for onset, when in sequencer.played:
        played.setdefault(onset, when)
    return played, state.acc_step_counter


def score_accompaniment(played, to_time, acc_onsets):
    """How late (or early) each accompaniment onset sounded, in milliseconds."""
    result = {"played": 100.0 * len(played) / max(len(acc_onsets), 1)}
    if not played:
        return result

    onsets = np.array(sorted(played))
    actual = np.array([played[o] for o in onsets])
    ideal = np.asarray(to_time(onsets), dtype=float)
    delta = 1000.0 * (actual - ideal)

    result["async50"] = float(np.percentile(np.abs(delta), 50))
    result["async95"] = float(np.percentile(np.abs(delta), 95))
    result["bias"] = float(np.median(delta))
    result["worst"] = float(np.max(np.abs(delta)))
    # A listener forgives a few ms; a fifth of a second is a wrong entry.
    result["late200"] = float(100.0 * np.mean(np.abs(delta) > 200))
    return result


HEADER = (
    f"{'follower':<17s}{'scenario':<9s}{'async50':>9s}{'async95':>9s}"
    f"{'bias':>8s}{'worst':>8s}{'>200ms':>8s}{'played':>8s}{'steps':>7s}"
)
LEGEND = """
async50/95  how far the accompaniment was from where it should have sounded,
            median and 95th percentile, in ms (absolute)
bias        median signed error: positive = the ACCompanion plays late
worst       largest single deviation, in ms
>200ms      share of accompaniment onsets off by more than a fifth of a second
played      share of the accompaniment score that sounded at all
steps       accompaniment steps the accompanist took"""


def main():
    parser = argparse.ArgumentParser(
        description="Measure what the ACCompanion plays, not just what it follows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--piece", help="piece name, e.g. badinerie")
    parser.add_argument("--solo", help="solo score file (overrides --piece)")
    parser.add_argument("--acc", help="accompaniment score file (overrides --piece)")
    parser.add_argument(
        "--followers", nargs="+", default=["PitchIOIHMM"],
        help="score followers to run. Default: PitchIOIHMM",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=["clean"],
        help="how the solo part is played: " + ", ".join(H.SCENARIOS) + ", or 'all'.",
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--bpm", type=float, default=130.0)
    parser.add_argument("--polling-period", type=float)
    parser.add_argument("--follower-kwargs", metavar="JSON")
    parser.add_argument(
        "--no-fermata", action="store_true",
        help="count through fermatas and free sections instead of waiting for "
        "the soloist at them, as the ACCompanion did before it could wait. "
        "Only changes anything on a piece that has them.",
    )
    parser.add_argument("--traceback", action="store_true")
    args = parser.parse_args()

    if args.solo and args.acc:
        solo_fn, acc_fn = args.solo, args.acc
    elif args.piece:
        solo_fn, acc_fn = H.find_piece(args.piece)
    else:
        parser.error("give either --piece or both --solo and --acc")

    scenarios = list(H.SCENARIOS) if "all" in args.scenarios else args.scenarios
    unknown = [s for s in scenarios if s not in H.SCENARIOS]
    if unknown:
        parser.error(f"unknown scenario(s) {unknown}")

    follower_kwargs = json.loads(args.follower_kwargs) if args.follower_kwargs else None

    print(f"piece:  {os.path.dirname(solo_fn)}")
    print(f"input:  solo score played at {args.bpm:g} bpm")
    print(f"scenarios: {', '.join(scenarios)}   seeds: {args.seeds}\n")
    print(HEADER)
    print("-" * len(HEADER))

    failures = []
    for follower in args.followers:
        polling_period = H.polling_period_for(follower, args.polling_period)
        for scenario_name in scenarios:
            scenario = H.SCENARIOS[scenario_name]
            seeds = [0] if not scenario.is_random else range(args.seeds)
            runs = []
            try:
                for seed in seeds:
                    messages, times, _, _, _ = H.render(
                        solo_fn, args.bpm, scenario, seed)
                    to_time, _ = H.beat_to_time_map_for(
                        solo_fn, args.bpm, scenario, seed)
                    with quiet():
                        accompanion = build(
                            follower, solo_fn, acc_fn, polling_period, args.bpm,
                            follower_kwargs=follower_kwargs,
                            fermata_kwargs={"enabled": not args.no_fermata,
                                            "verbose": False})
                    if getattr(accompanion, "event_based_input", False):
                        frames = H.event_frames(messages, times)
                    else:
                        frames = H.framed(messages, times, polling_period)
                    played, steps = run_once(accompanion, frames, to_time)
                    acc_onsets = accompanion.acc_score.unique_onsets
                    r = score_accompaniment(played, to_time, acc_onsets)
                    r["steps"] = steps
                    runs.append(r)
            except Exception as exc:  # noqa: BLE001 - a failure is a result
                if args.traceback:
                    import traceback
                    traceback.print_exc()
                print(f"{follower:<17s}{scenario_name:<9s}FAILED: "
                      f"{type(exc).__name__}: {exc}")
                failures.append((follower, scenario_name))
                continue

            r = H.aggregate(runs)

            def num(key, fmt):
                v = r.get(key)
                return format(v, fmt) if v is not None and np.isfinite(v) else "-"

            print(f"{follower:<17s}{scenario_name:<9s}"
                  f"{num('async50', '9.1f')}{num('async95', '9.1f')}"
                  f"{num('bias', '8.1f')}{num('worst', '8.0f')}"
                  f"{num('late200', '8.1f')}{num('played', '8.1f')}"
                  f"{int(round(r.get('steps', 0))):7d}", flush=True)

    print(LEGEND)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
