#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Write pyanoduo's ``sync.csv`` into every take of ``aura-data``.

pyanoduo's MIDI-conditioned training reads, per take, a ``sync.csv`` of
``filename,offset_s`` rows placing each file on the recording's shared
timeline: a MIDI file's local time starts at its first performed note, the
video's at its first frame, and ``shared = offset_s + local``.

This writes the offsets that make the shared timeline the MIDI files' own
clock: each MIDI file's offset is the time of its first note in that file,
and the video's offset is ``--video-offset`` (0 by default). That is the
assumption the rest of ``aura-data`` rests on -- the MIDI recorder and the
camera started together -- checked on D05/B1_T1_L1 to within about half a
second from the motion itself. When a take's video offset has been measured
(the frame of the first key press against the first MIDI onset), pass it
with ``--video-offset`` for that take, or edit the file.

Usage
-----
    python bin/write_aura_sync.py --data ~/datasets/aura-data
    python bin/write_aura_sync.py --data ~/datasets/aura-data --takes D05/B1_T1_L1 --video-offset 0.1
"""
import argparse
import csv
import fnmatch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import partitura as pt

from aura_layout import detect_layout, read_metadata


def first_onset(midi_fn):
    notes = pt.load_performance_midi(midi_fn, merge_tracks=True, quiet=True)[0].note_array()
    return float(notes["onset_sec"].min())


def main():
    parser = argparse.ArgumentParser(
        description="Write sync.csv files for pyanoduo into the takes of aura-data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True, help="the aura-data folder (nested layout)")
    parser.add_argument("--takes", nargs="+", metavar="GLOB", help="only takes matching")
    parser.add_argument("--video-offset", type=float, default=0.0,
                        help="shared time of the videos' first frame, in seconds. Default: 0")
    parser.add_argument("--camera", default="top")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = os.path.abspath(args.data)
    if detect_layout(data) != "nested":
        raise SystemExit("sync.csv lives in the take folders: convert to the nested layout first")
    parent = os.path.dirname(data)
    rows, _ = read_metadata(data)
    takes = {}
    for row in rows:
        takes.setdefault((row["duo"], row["take"]), []).append(row)

    written = 0
    for (duo, take), players in sorted(takes.items()):
        key = f"{duo}/{take}"
        if args.takes and not any(fnmatch.fnmatch(key, p) for p in args.takes):
            continue
        folder = os.path.join(data, duo, take)
        out = os.path.join(folder, "sync.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"have {key}")
            continue
        entries = [(f"video_{p}_{args.camera}.mp4", args.video_offset) for p in ("p1", "p2")]
        for row in sorted(players, key=lambda r: r["player"]):
            midi_fn = os.path.join(parent, row["midi"])
            entries.append((os.path.basename(midi_fn), first_onset(midi_fn)))
        with open(out, "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["filename", "offset_s"])
            for name, offset in entries:
                writer.writerow([name, f"{offset:.6f}"])
        written += 1
        print(f"{key}: " + ", ".join(f"{n} {o:.3f}" for n, o in entries))
    print(f"\n{written} sync.csv written")


if __name__ == "__main__":
    main()
