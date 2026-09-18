#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Avatar training input from recorded performances of the ACCompanion's part.

Live, the avatar is driven by two streams the ACCompanion produces while it
plays (see `accompanion.avatar`): the MIDI messages of its own part, each
tagged with its score note, and its score position and tempo on every frame.
This writes those same streams, and the per-frame features derived from
them, for every recording in `aura-data` of the part the avatar will play --
the secondo of the badinerie, by default -- so that the motion policy is
trained on exactly the input it will be run on.

Where each stream comes from offline:

notes
    The player's MIDI file, with each note's score note taken from the match
    file. Notes the alignment could not place are kept as key presses with
    no score note (the ACCompanion never plays such a note, but the human
    did, and their hands moved for it). Score notes the player left out are
    kept out of the lookahead, so that every note it announces is one that
    gets played, as is always the case live (``--keep-deleted`` turns this
    off). See `accompanion.avatar` for the reasoning.
frames
    A position/tempo estimate, chosen with ``--position``:

    ``online-primo`` (default)
        What the ACCompanion estimated while following the *other* player
        of the same take -- the primo -- as it will live, mistakes and
        lag included. From ``tempo_curves/<duo>/<take>/<primo>.npz``.
    ``offline-self``
        The ground-truth position of this very performance, from its own
        offline alignment. Cleaner, but not what the avatar will get live:
        live, the position is the ACCompanion's estimate of the *primo*, and
        the ACCompanion's notes follow that estimate by construction; a
        human secondo does not follow the primo that exactly.
    ``offline-primo``
        The ground-truth position of the primo of the same take.

Output, per performance, in the folder's layout (``avatar/<name>.npz`` or
``<duo>/<take>/avatar_<player>.npz``, see `aura_layout`):

    note_events, frame_events    the streams (accompanion.avatar dtypes)
    score_notes                  the part, for the lookahead
    skipped_note_ids             score notes this performance never plays
    timestamps, features         features on a fixed clock (``--fps``)
    feature_names                one name per feature column
    position_source, lookahead, recency_tau, horizon_sec, beats_per_bar

The motion side can use ``features`` directly when its frame clock matches
(pyanoduo rebases video to ``frame_index / fps``), or rebuild them for its
own timestamps with ``AvatarFeaturizer(score_notes,
skipped_ids=skipped_note_ids).run(note_events, frame_events, timestamps)``.

Usage
-----
    python bin/build_avatar_features.py --data ~/datasets/aura-data
    python bin/build_avatar_features.py --data ~/datasets/aura-data \\
        --position offline-self --name avatar-offline-self
"""
import argparse
import json
import os
import shutil
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "bin"))

import numpy as np
import partitura as pt

from accompanion.avatar import (
    FRAME_EVENT_DTYPE,
    NOTE_EVENT_DTYPE,
    AvatarFeaturizer,
    score_notes_from_note_array,
)
from aura_layout import (
    collection_path,
    convert,
    detect_layout,
    index_path,
    performance_paths,
    read_metadata,
)
from extract_tempo_curves import warp_table

POSITION_SOURCES = ("online-primo", "offline-self", "offline-primo")

#: Beats per bar of each piece, for the bar phase feature.
BEATS_PER_BAR = {"Badinerie": 4.0}


def note_events_from_performance(midi_fn, match_fn, score_notes):
    """`NoteEvent` rows for a recorded performance.

    One ``on`` and one ``off`` per performed note, in the MIDI file's clock;
    the score note comes from the match file where the alignment placed the
    note.
    """
    ppart = pt.load_performance_midi(midi_fn)[0]
    _, alignment = pt.load_match(match_fn, create_score=False)
    score_of = {
        a["performance_id"]: a["score_id"] for a in alignment if a["label"] == "match"
    }
    by_id = {nid: i for i, nid in enumerate(score_notes["note_id"])}

    skipped = sorted({a["score_id"] for a in alignment if a["label"] == "deletion"})

    rows = []
    for note in ppart.note_array():
        pid = str(note["id"])
        sid = score_of.get(pid)
        index = by_id.get(sid) if sid is not None else None
        if index is not None:
            onset_beat = float(score_notes["onset_beat"][index])
            duration_beat = float(score_notes["duration_beat"][index])
            note_id = str(score_notes["note_id"][index])
        else:
            onset_beat, duration_beat, note_id = np.nan, np.nan, ""
        on = float(note["onset_sec"])
        off = on + float(note["duration_sec"])
        rows.append((on, True, int(note["pitch"]), int(note["velocity"]),
                     onset_beat, duration_beat, note_id))
        rows.append((off, False, int(note["pitch"]), 0,
                     onset_beat, duration_beat, note_id))
    events = np.array(rows, dtype=NOTE_EVENT_DTYPE)
    return np.sort(events, order=["time_sec", "on"]), skipped, len(score_of), len(ppart.notes)


def frame_events_from_online(curves_npz):
    """`FrameEvent` rows from the ACCompanion's own per-frame record."""
    frames = np.load(curves_npz)["online_frames"]
    events = np.zeros(len(frames), dtype=FRAME_EVENT_DTYPE)
    events["time_sec"] = frames["time_sec"]
    events["position_beat"] = frames["expected_position_beat"]
    events["beat_period"] = frames["beat_period"]
    events["waiting"] = frames["waiting"]
    return events


def frame_events_from_offline(curves_npz, period=0.01, margin=1.0):
    """`FrameEvent` rows sampled from an offline beat map.

    The position follows the hold-aware warp of the knots, the beat period
    is interpolated between knots, and ``waiting`` is set inside each hold.
    """
    knots = np.load(curves_npz)["offline_knots"]
    seconds, beats = warp_table(knots)
    t0, t1 = seconds[0] - margin, seconds[-1] + margin
    times = np.arange(t0, t1, period)
    events = np.zeros(len(times), dtype=FRAME_EVENT_DTYPE)
    events["time_sec"] = times
    events["position_beat"] = np.interp(times, seconds, beats)
    events["beat_period"] = np.interp(
        times, knots["perf_onset_sec"], knots["beat_period"]
    )
    waiting = np.zeros(len(times), dtype=bool)
    for knot in knots[knots["is_fermata"] & (knots["hold_sec"] > 0)]:
        start = knot["perf_onset_sec"]
        waiting |= (times >= start) & (times < start + knot["hold_sec"])
    events["waiting"] = waiting
    return events


def main():
    parser = argparse.ArgumentParser(
        description="Write the avatar's input streams and features per recording.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True, help="the aura-data folder")
    parser.add_argument("--name", default="avatar",
                        help="collection name, e.g. 'avatar-offline-self' for a "
                        "variant. Default: avatar")
    parser.add_argument("--keep-deleted", action="store_true",
                        help="keep score notes the player left out in the lookahead")
    parser.add_argument("--piece", default="Badinerie")
    parser.add_argument("--part", default="secondo", choices=["primo", "secondo"],
                        help="the part the avatar plays. Default: secondo")
    parser.add_argument("--position", choices=POSITION_SOURCES, default="online-primo",
                        help="which position/tempo estimate stands in for the "
                        "live one. Default: online-primo")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="clock of the precomputed features. Default: 30")
    parser.add_argument("--lookahead", type=int, default=8)
    parser.add_argument("--recency-tau", type=float, default=0.25)
    parser.add_argument("--horizon-sec", type=float, default=5.0)
    parser.add_argument("--takes", nargs="+", metavar="GLOB",
                        help="only takes matching, e.g. 'D05/*'")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    data = os.path.abspath(args.data)
    parent = os.path.dirname(data)
    layout = detect_layout(data)

    rows, _ = read_metadata(data)
    by_take = {}
    for row in rows:
        by_take.setdefault((row["duo"], row["take"]), {})[row["part"]] = row
    selected = [
        (key, parts) for key, parts in sorted(by_take.items())
        if parts.get(args.part, {}).get("piece") == args.piece
        and (not args.takes or any(
            __import__("fnmatch").fnmatch("/".join(key), p) for p in args.takes))
    ]
    if not selected:
        parser.error(f"no {args.piece} {args.part} performances in {data}")

    score_fn = os.path.join(parent, selected[0][1][args.part]["score"])
    score_na = pt.load_score(score_fn)[0].note_array(include_grace_notes=True)
    score_notes = score_notes_from_note_array(score_na)
    featurizer = AvatarFeaturizer(score_notes, lookahead=args.lookahead)  # for dim/names
    print(f"{len(selected)} {args.piece} {args.part} performance(s); position from "
          f"{args.position}; {featurizer.dim} features at {args.fps:g} fps -> "
          f"'{args.name}' in the {layout} layout of {data}\n")

    index = []
    started = time.perf_counter()
    for (duo, take), parts in selected:
        me = parts[args.part]
        other = parts["primo" if args.part == "secondo" else "secondo"]
        key = (me["piece"], me["part"], duo, take, me["player"])
        name = "_".join(key)

        def curve_of(row):
            return os.path.join(data, performance_paths(
                layout, row["piece"], row["part"], duo, take, row["player"]
            )["tempo_curves"])

        note_events, skipped, n_scored, n_notes = note_events_from_performance(
            os.path.join(parent, me["midi"]),
            os.path.join(parent, me["match"]),
            score_notes,
        )
        if args.keep_deleted:
            skipped = []
        if args.position == "online-primo":
            source = curve_of(parts["primo"])
            frame_events = frame_events_from_online(source)
        elif args.position == "offline-self":
            source = curve_of(me)
            frame_events = frame_events_from_offline(source)
        else:
            source = curve_of(parts["primo"])
            frame_events = frame_events_from_offline(source)

        featurizer = AvatarFeaturizer(
            score_notes,
            lookahead=args.lookahead,
            recency_tau=args.recency_tau,
            horizon_sec=args.horizon_sec,
            beats_per_bar=BEATS_PER_BAR.get(args.piece, 4.0),
            skipped_ids=skipped,
        )
        end = max(note_events["time_sec"].max(), frame_events["time_sec"].max()) + 1.0
        timestamps = np.arange(0.0, end, 1.0 / args.fps)
        features = featurizer.run(note_events, frame_events, timestamps)

        out_fn = os.path.join(data, collection_path(layout, args.name, *key))
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        np.savez(
            out_fn,
            note_events=note_events,
            frame_events=frame_events,
            score_notes=score_notes,
            skipped_note_ids=np.array(skipped, dtype="U16"),
            timestamps=timestamps,
            features=features.astype(np.float32),
            feature_names=np.array(featurizer.feature_names),
            position_source=np.array(args.position),
            position_file=np.array(os.path.relpath(source, parent)),
            lookahead=args.lookahead,
            recency_tau=args.recency_tau,
            horizon_sec=args.horizon_sec,
            beats_per_bar=featurizer.beats_per_bar,
        )
        info = {
            "name": name, "file": os.path.relpath(out_fn, parent),
            "duo": duo, "take": take, "player": me["player"], "partner": other["player"],
            "notes": n_notes, "notes_with_score_note": n_scored,
            "score_notes_skipped": len(skipped),
            "frames": int(len(timestamps)), "position_source": args.position,
            "position_file": os.path.relpath(source, parent),
        }
        index.append(info)
        print(f"  {name}: {n_scored}/{n_notes} notes scored, {len(skipped)} score notes "
              f"skipped, {len(timestamps)} frames")

    index_fn = os.path.join(data, index_path(layout, args.name))
    os.makedirs(os.path.dirname(index_fn), exist_ok=True)
    with open(index_fn, "w") as f:
        json.dump({
            "piece": args.piece, "part": args.part, "position_source": args.position,
            "keep_deleted": bool(args.keep_deleted),
            "fps": args.fps, "feature_dim": featurizer.dim,
            "feature_names": featurizer.feature_names,
            "performances": index,
        }, f, indent=2)
    # Fill in the metadata's path columns for what now exists, and put the
    # description of these files next to them.
    convert(data, layout, verbose=False)
    shutil.copyfile(
        os.path.join(REPO_ROOT, "docs", "avatar_input.md"),
        os.path.join(data, "AVATAR.md"),
    )
    print(f"\n{len(index)} performances in {time.perf_counter() - started:.0f}s -> {index_fn}")


if __name__ == "__main__":
    main()
