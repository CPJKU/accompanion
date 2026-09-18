#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run pyanoduo's MIDI-conditioned training (the ``dataset-midi-train`` branch)
on ``aura-data`` as it is.

Two checks:

1. pyanoduo's own score audit on every take: its Matchmaker clock
   (``PitchIOIHMM`` + ``KalmanTempoModel``) follows the primo of each take
   and must reach 90% of the score. This is the follower pyanoduo trains and
   replays with, so this is the number that matters for it.
2. ``run_score_experiment`` on the takes that have motion, with ``aura-data``
   as both ``root`` and ``motion_root`` and its ``score/`` MIDI score --
   a short run, to prove the data contract, not to train a policy.

Needs pyanoduo's environment.

Usage
-----
    python bin/test_pyanoduo_midi.py --data ~/datasets/aura-data --pyanoduo ~/cp/pyanoduo
    python bin/test_pyanoduo_midi.py ... --skip-audit --epochs 50 --output runs/aura_smoke
"""
import argparse
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="the aura-data folder (nested layout)")
    parser.add_argument("--pyanoduo", required=True, help="the pyanoduo repository")
    parser.add_argument("--duos", nargs="+", default=["D05"])
    parser.add_argument("--recordings", nargs="+", default=["B1_T1_L1"],
                        help="takes with motion, for the training run")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--distill-epochs", type=int, default=5)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--output", help="run folder. Default: a temporary one")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.pyanoduo))
    from pyanoduo.modules.music import load_score
    from pyanoduo.modules.score_data import ScoreDataConfig, check_score
    from pyanoduo.modules.score_training import ScoreRunConfig, run_score_experiment

    data = Path(args.data).resolve()
    score_fn = data / "score" / "P1-Badinerie_of_the_Mouse_Jester_Primo.mid"
    score = load_score(score_fn)

    if not args.skip_audit:
        takes = sorted(p for p in data.glob("D*/B1_*") if (p / "sync.csv").is_file())
        print(f"score audit with pyanoduo's clock on {len(takes)} takes:")
        ok, failed = [], []
        started = time.perf_counter()
        for take in takes:
            try:
                audit = check_score(take, score)
                ok.append(audit)
                print(f"  {audit['recording']}: primo {audit['primo_participant']}, followed to beat "
                      f"{audit['followed_end_beat']:g} of {audit['score_last_onset']:g} "
                      f"(coverage {audit['coverage']:.2f}), {audit['duration_s']:.0f}s")
            except ValueError as error:
                failed.append(str(error))
                print(f"  FAILED: {error}")
        print(f"  {len(ok)} passed, {len(failed)} failed, {time.perf_counter() - started:.0f}s\n")

    output = args.output or os.path.join(tempfile.mkdtemp(prefix="aura_midi_"), "run")
    config = ScoreRunConfig.from_toml(os.path.join(args.pyanoduo, "configs", "midi.toml"))
    config = replace(
        config,
        output=output,
        assets=os.path.join(os.path.abspath(args.pyanoduo), "MHR", "assets"),
        data=replace(
            config.data, root=str(data), motion_root=str(data), score=str(score_fn),
            duos=tuple(args.duos), recordings=tuple(args.recordings), held_out=(),
        ),
        training=replace(config.training, epochs=args.epochs,
                         distill_epochs=args.distill_epochs, log_every=max(1, args.epochs // 4)),
        preview=replace(config.preview, seconds=4.0, frames=30, size=320, mesh=False),
    )
    print(f"run_score_experiment on {args.duos} {args.recordings}, {args.epochs} epochs -> {output}")
    started = time.perf_counter()
    result = run_score_experiment(config, progress=True)
    print(f"\ndone in {time.perf_counter() - started:.0f}s: {result}")
    for name in sorted(os.listdir(output)):
        print("  ", name)
    if not args.output:
        shutil.rmtree(os.path.dirname(output), ignore_errors=True)


if __name__ == "__main__":
    main()
