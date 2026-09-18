#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the avatar input through pyanoduo end to end, on the recordings that
have motion.

Three checks, in order:

1. pyanoduo loads the recording pair from ``aura-data`` and the
   `AvatarConditioner` returns one feature row per motion frame, finite, and
   identical to the features stored at 30 fps where the clocks coincide.
2. pyanoduo's smoke training runs with ``midi_enabled``: the policy takes the
   224 avatar features next to the partner's motion, trains, distills and
   evaluates.
3. The trained policy is driven frame by frame through pyanoduo's
   `RealtimeController` with the same features -- the shape of the live loop.

Needs pyanoduo's environment.

Usage
-----
    python bin/test_avatar_pipeline.py --data ~/datasets/aura-data --pyanoduo ~/cp/pyanoduo
    python bin/test_avatar_pipeline.py ... --collection avatar-offline-self --epochs 5
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np

from accompanion.avatar import AvatarConditioner


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", required=True, help="the aura-data folder")
    parser.add_argument("--pyanoduo", required=True, help="the pyanoduo repository")
    parser.add_argument("--collection", default="avatar")
    parser.add_argument("--duos", nargs="+", default=["D05"])
    parser.add_argument("--recordings", nargs="+", default=["B1_T1_L1"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--output", help="run folder. Default: a temporary one")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.pyanoduo))
    from pyanoduo.modules.control import RealtimeController
    from pyanoduo.modules.dataset import MotionDataset, load_recordings
    from pyanoduo.modules.features import MotionFeatures
    from pyanoduo.modules.mhr import MHRRig
    from pyanoduo.modules.training import RunConfig, load_policy, run_experiment

    data = os.path.abspath(args.data)
    assets = os.path.join(os.path.abspath(args.pyanoduo), "MHR", "assets")

    # 1. recordings and features -------------------------------------------
    recordings = load_recordings(data, duos=tuple(args.duos), recordings=tuple(args.recordings))
    conditioner = AvatarConditioner(data, args.collection)
    print(f"{len(recordings)} recording pair(s); conditioner '{args.collection}', "
          f"{conditioner.feature_dim} features\n")
    for rec in recordings:
        meta = rec.metadata
        X = conditioner.features(meta, rec.timestamps)
        stored = np.load(conditioner.file_for(meta))
        n = min(len(X), len(stored["features"]))
        same = np.allclose(X[:n], stored["features"][:n], atol=1e-5)
        print(f"  {rec.key}: target {meta.participant}, partner {'p1' if meta.participant == 'p2' else 'p2'}, "
              f"{len(rec.timestamps)} frames {rec.timestamps[0]:.2f}-{rec.timestamps[-1]:.2f}s -> "
              f"features {X.shape}, finite {np.isfinite(X).all()}, equal to stored 30 fps features: {same}")
        col = {name: i for i, name in enumerate(conditioner.feature_names)}
        active = X[:, col["position_beat"]] > 0
        print(f"    position {X[active, col['position_beat']].min():.1f}-{X[:, col['position_beat']].max():.1f} beats, "
              f"tempo {60 / X[active, col['beat_period_sec']].max():.0f}-{60 / X[active, col['beat_period_sec']].min():.0f} bpm, "
              f"frames waiting {int(X[:, col['waiting']].sum())}, frames with a key down "
              f"{int((X[:, :88].sum(1) > 0).sum())}")

    # 2. training ------------------------------------------------------------
    rig = MHRRig(assets)
    config = RunConfig.from_toml(os.path.join(args.pyanoduo, "configs", "smoke.toml"))
    output = args.output or os.path.join(tempfile.mkdtemp(prefix="avatar_smoke_"), "run")
    from dataclasses import replace
    config = replace(
        config,
        output=output,
        assets=assets,
        data=replace(config.data, root=data, duos=tuple(args.duos),
                     recordings=tuple(args.recordings), midi_enabled=True),
        training=replace(config.training, epochs=args.epochs),
        distillation=replace(config.distillation, epochs=args.epochs),
    )
    features = MotionFeatures(rig, config.features)
    dataset = MotionDataset(recordings, features, history=config.data.history,
                            max_windows_per_recording=config.data.max_windows_per_recording,
                            midi=conditioner)
    batch = dataset[0]
    print(f"\ndataset: {len(dataset)} windows; one item: "
          + ", ".join(f"{k} {tuple(v.shape)}" for k, v in batch.items()))

    print(f"\ntraining {args.epochs} epoch(s) with midi_enabled -> {output}")
    started = time.perf_counter()
    result = run_experiment(config, midi=conditioner, progress=False)
    print(f"done in {time.perf_counter() - started:.0f}s: {result}")
    rows = open(os.path.join(output, "training.csv")).read().strip().splitlines()
    print("  training.csv:", rows[0]); [print("   ", r) for r in rows[1:4]]
    print("  files:", sorted(os.listdir(output)))

    # 3. the live loop -------------------------------------------------------
    policy, saved, _ = load_policy(os.path.join(output, "flow.pt"), device="cpu")
    controller = RealtimeController(policy, features, history=saved.data.history)
    rec = recordings[0]
    controller.reset(seed=0, initial_pose=rec.target[0])
    X = conditioner.features(rec.metadata, rec.timestamps)
    poses = []
    started = time.perf_counter()
    for t in range(min(90, len(rec.timestamps))):
        poses.append(controller.step(rec.partner[t], rec.context, midi=X[t]))
    per_frame = (time.perf_counter() - started) / len(poses) * 1000
    poses = np.array(poses)
    print(f"\nlive loop: {len(poses)} frames driven through RealtimeController with the avatar "
          f"features, {per_frame:.1f} ms/frame; pose {poses.shape}, finite {np.isfinite(poses).all()}, "
          f"mean |pose - human| {np.abs(poses - rec.target[:len(poses)]).mean():.3f}")
    if not args.output:
        shutil.rmtree(os.path.dirname(output), ignore_errors=True)


if __name__ == "__main__":
    main()
