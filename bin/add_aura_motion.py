#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Add the players' motion to ``aura-data``, prepared with pyanoduo.

The motion capture arrives as SAM 3D Body pickles, one per player and take
(``<duo>/<take>/video_p1_top.pkl``, half a gigabyte each). pyanoduo turns
each into its canonical ``motion.npz`` -- MHR pose parameters per video
frame, cleaned and repaired -- plus a summary, per-frame metrics and a
diagnostics plot. This runs ``pyanoduo prepare`` on every pickle it finds and
files the result as the take's ``video_<player>_top/`` folder (or
``motion/<name>/`` in the flat layout), leaving the pickles where they are.
The folder is named as pyanoduo's own motion cache names it, so that
``aura-data`` serves as pyanoduo's ``motion_root`` directly.

The clock of the motion is ``frame_index / fps`` from the first video frame,
taken to be the MIDI file's clock as well; see the README.

Needs pyanoduo's environment (``pyanoduo`` on the path, or ``--pyanoduo``).

Usage
-----
    python bin/add_aura_motion.py --data ~/datasets/aura-data \\
        --pickles ~/cp/pyanoduo/dataset/D02-D06 \\
        --pyanoduo ~/miniforge3/envs/pyanoduo/bin/pyanoduo
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from aura_layout import convert, detect_layout, performance_paths, read_metadata

VIDEO = re.compile(r"video_(p\d)_(\w+)\.pkl$")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the players' motion with pyanoduo and file it in aura-data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True, help="the aura-data folder")
    parser.add_argument("--pickles", required=True,
                        help="folder holding <duo>/<take>/video_p?_top.pkl")
    parser.add_argument("--pyanoduo", default="pyanoduo", help="the pyanoduo executable")
    parser.add_argument("--assets", help="pyanoduo's MHR assets folder, if not its default")
    parser.add_argument("--camera", default="top")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data = os.path.abspath(args.data)
    layout = detect_layout(data)
    rows, _ = read_metadata(data)
    by_key = {(r["duo"], r["take"], r["player"]): r for r in rows}

    pickles = sorted(glob.glob(os.path.join(args.pickles, "**", "video_*.pkl"), recursive=True))
    done = skipped = 0
    for pkl in pickles:
        m = VIDEO.search(os.path.basename(pkl))
        duo, take = pkl.split(os.sep)[-3:-1]
        if not m or m.group(2) != args.camera:
            continue
        player = m.group(1)
        row = by_key.get((duo, take, player))
        if row is None:
            print(f"skip {duo}/{take} {player}: not in metadata-aura.csv")
            skipped += 1
            continue
        target = os.path.join(data, performance_paths(
            layout, row["piece"], row["part"], duo, take, player)["motion"])
        if os.path.exists(os.path.join(target, "motion.npz")) and not args.overwrite:
            print(f"have {duo}/{take} {player}")
            continue

        print(f"prepare {duo}/{take} {player} ({row['part']}) from {pkl}")
        with tempfile.TemporaryDirectory() as tmp:
            command = [args.pyanoduo]
            if args.assets:
                command += ["--assets", args.assets]
            command += ["prepare", pkl, tmp]
            subprocess.run(command, check=True)
            produced = os.path.join(tmp, os.path.splitext(os.path.basename(pkl))[0])
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.move(produced, target)
        done += 1

    convert(data, layout, verbose=False)
    print(f"\n{done} prepared, {skipped} skipped; metadata updated")


if __name__ == "__main__":
    main()
