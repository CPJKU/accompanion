#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Package the duo recordings as an aligned dataset, laid out like ``data/asap``.

Every performance in the duo dataset is aligned to the part it plays with
parangonar, and the performance MIDI, the alignment and the score are written
into one self-describing folder, in either of the layouts `aura_layout`
knows: ``nested`` (one folder per take, as the recordings come; the default)
or ``flat`` (one folder per kind of file, as in ``data/asap``). Convert
between them at any time with ``bin/aura_layout.py --to``.

``p1`` / ``p2`` names the MIDI file a performance was recorded to, and the
metadata's ``part`` column says which part it holds: the players swap parts
between takes, so the file name alone does not tell.

The ``.match`` files are what the ACCompanion, partitura and parangonar read;
the ``parangonada/`` folders are the same alignments in the CSV layout the
parangonada editor loads, for going over them by hand. See the README the
script writes for how to bring an edited alignment back into a ``.match``.

The alignment itself comes from `extract_tempo_curves`, so the match files
here and the offline tempo curves there agree note for note.

Usage
-----
    python bin/build_aura_dataset.py --dataset ~/datasets/aura/midi_dataset --out ~/datasets/aura-data --takes '*/B1_*'
    python bin/build_aura_dataset.py --dataset ... --out ... --takes '*/B1_*' --layout flat
"""
import argparse
import os
import shutil
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "bin"))

import numpy as np
import partitura as pt

from aura_layout import (
    LAYOUT_SECTIONS,
    LAYOUTS,
    PATH_COLUMNS,
    performance_paths,
    score_path,
    update_readme,
    write_metadata,
)
from extract_tempo_curves import (
    DEFAULT_BLOCK_PIECES,
    ROLES,
    Piece,
    assign_roles,
    discover_takes,
    make_matcher,
    parse_block_pieces,
    piece_for_take,
)

#: File-name stem per piece directory, and the title to record for it.
PIECE_NAMES = {
    "badinerie": ("Badinerie", "Badinerie of the Mouse Jester"),
    "fanfare": ("Fanfare", "The Blue Fanfare of the Empress Cat"),
}

#: What the take's suffix says about how the duo was instructed to play.
CONDITIONS = {
    "L1": ("leader-follower", "p1"),
    "L2": ("leader-follower", "p2"),
    "E": ("engage", ""),
    "D": ("disengage", ""),
}

METADATA_COLUMNS = list(PATH_COLUMNS) + [
    "piece", "title", "part", "duo", "take", "block", "trial",
    "condition", "condition_name", "leader", "player",
    "performed_notes", "score_notes", "matched", "inserted", "deleted",
    "source_midi",
]

README = """\
# aura-data

Piano duo recordings aligned to their scores: one performance MIDI file per
player and take, a note-level alignment of it to the part that player was
playing, and the score of that part.

{layout_section}
## Naming

- `<Piece>` -- {pieces_line}
- `<part>` -- `primo` or `secondo`: the part this player played in this take
- `<duo>` -- the pair of players, `D02` ... `D12`. `D07` is missing: something
  went wrong during its recording.
- `<take>` -- `<block>_<trial>_<condition>`: {blocks_line}; trials `T1` ...
  `T4`; conditions `L1`/`L2` (leader-follower, with `p1`/`p2` leading), `E`
  (engage), `D` (disengage)
- `<player>` -- `p1` or `p2`: the MIDI file (`midi_p1.mid`, `midi_p2.mid`)
  the performance was recorded to, as in the motion capture files
  (`video_p1_top.pkl`).

**Which file holds which part is not fixed.** Aligning every file against
both parts shows that in every `L1` take `midi_p1.mid` holds the primo and
`midi_p2.mid` the secondo, and in every `L2` take it is the other way round:
`midi_p1.mid` holds the secondo. The `part` column of the metadata records
what each file actually contains; do not infer it from the file name. Whether
the players changed seats or the recording channels were swapped in the `L2`
takes -- which decides whose motion `video_p1_top.pkl` shows there -- cannot
be told from the MIDI.

Paths in `metadata-aura.csv` are relative to this folder's parent, as in
`asap/metadata-asap.csv`.

## How the alignments were made

`parangonar.DualDTWNoteMatcher` over the performance's note array and the
part's (grace notes included), written with `partitura.save_match`. The
scores are the ACCompanion's `accompanion_pieces/simple_pieces/<piece>/`
files. Built by `bin/build_aura_dataset.py` in the ACCompanion repository.

The recordings are not perfect and neither are the alignments: `matched`,
`inserted` and `deleted` in the metadata count the performed notes matched to
a score note, the performed notes the matcher could not place, and the score
notes it found no performance of.

## Tempo curves and avatar input

The tempo curve files hold, per take and player, the performance-time to
score-time map two ways: `offline_knots` from the alignment above, and the
`online_*` arrays from the ACCompanion following the performance with the
code that plays live. The summary next to them compares the two. Made by
`bin/extract_tempo_curves.py`; its docstring lists every column.

The avatar files hold, per performance of the part the avatar plays (the
badinerie secondo), the two event streams the ACCompanion produces live for
its own part -- `note_events` (its MIDI messages, each with its score note)
and `frame_events` (score position, tempo, fermata hold per frame) -- and
the per-frame `features` computed from them at 30 fps, with `feature_names`.
`avatar` uses the ACCompanion's own position estimate from following the
primo of the same take; `avatar-offline-self` the ground-truth position of
the secondo performance instead. See `accompanion/avatar.py` for the streams
and the features, and `bin/build_avatar_features.py` for how they are made
from a recording.

The `video_<player>_top/` folders hold each player's motion as pyanoduo
prepares it from the SAM 3D Body video estimate: `motion.npz` with MHR pose
parameters per video frame (30 fps), its `summary.json`, per-frame
`metrics.csv` and a `diagnostics.png`. Made by `bin/add_aura_motion.py`,
which runs `pyanoduo prepare` on the pickles and files the results here
under the names pyanoduo's own motion cache uses, so this folder serves as
its `motion_root`. Each take's `sync.csv` is pyanoduo's placement of the
files on a shared timeline (`bin/write_aura_sync.py`). `AVATAR.md` describes
the avatar files and how pyanoduo consumes all of this.

Times in all of these are seconds in the MIDI file's clock, taken to be the
motion capture's clock as well (`frame_index / fps` in pyanoduo). That has
been checked on one take to within about half a second; a frame-exact check
needs the raw video.

## Reviewing an alignment in parangonada

Open the performance's `parangonada` folder in parangonada, edit, and
export `align.csv`. To write the result back as a match file:

```python
import partitura as pt
from partitura.io.importparangonada import load_parangonada_alignment
alignment = load_parangonada_alignment("D05/B1_T1_L1/parangonada_p1/align.csv")
pt.save_match(
    alignment,
    pt.load_performance_midi("D05/B1_T1_L1/midi_p1.mid"),
    pt.load_score("score/Badinerie_primo_xml_score.musicxml"),
    out="D05/B1_T1_L1/midi_p1.match",
)
```
(paths as in the nested layout; in the flat one use the `<name>` files.)
"""


def performance_name(piece_stem, role, duo, take, player):
    return f"{piece_stem}_{role}_{duo}_{take}_{player}"


def main():
    parser = argparse.ArgumentParser(
        description="Package the duo recordings as an aligned dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True,
                        help="root holding <duo>/<take>/midi_p*.mid")
    parser.add_argument("--out", required=True, help="the aura-data folder to write")
    parser.add_argument("--takes", nargs="+", metavar="GLOB",
                        help="only takes matching, e.g. '*/B1_*'")
    parser.add_argument(
        "--block-piece", nargs="+", metavar="BLOCK=PIECE", dest="block_pieces",
        help="which piece a take block records. Default: "
        + " ".join(f"{k}={v}" for k, v in DEFAULT_BLOCK_PIECES.items()),
    )
    parser.add_argument("--layout", choices=LAYOUTS, default="nested",
                        help="folder layout to write, see bin/aura_layout.py. "
                        "Default: nested")
    parser.add_argument("--matcher", choices=["dualdtw", "automatic"], default="dualdtw")
    parser.add_argument("--overwrite", action="store_true",
                        help="redo performances whose match file exists")
    args = parser.parse_args()
    args.block_pieces = parse_block_pieces(args.block_pieces)
    sys.stdout.reconfigure(line_buffering=True)

    takes = discover_takes(args.dataset, args.takes)
    if not takes:
        parser.error(f"no takes found under {args.dataset}")

    out = os.path.abspath(args.out)
    prefix = os.path.basename(out)
    os.makedirs(os.path.join(out, "score"), exist_ok=True)

    matcher = make_matcher(args.matcher)
    pieces = {}
    rows = []
    started = time.perf_counter()
    print(f"{len(takes)} take(s) -> {out}\n")

    for i, (duo, take, files) in enumerate(takes, 1):
        piece_dir = piece_for_take(take, args.block_pieces)
        if piece_dir not in pieces:
            pieces[piece_dir] = Piece(piece_dir)
            stem = PIECE_NAMES.get(piece_dir, (piece_dir.capitalize(), piece_dir))[0]
            for role in ROLES:
                shutil.copyfile(
                    pieces[piece_dir].files[role],
                    os.path.join(out, score_path(stem, role)),
                )
        piece = pieces[piece_dir]
        stem, title = PIECE_NAMES.get(piece_dir, (piece_dir.capitalize(), piece_dir))

        performances = {
            player: pt.load_performance_midi(fn)[0] for player, fn in files.items()
        }
        roles, margin = assign_roles(
            {player: part.note_array() for player, part in performances.items()},
            piece,
        )
        block, trial, condition = take.split("_", 2)
        condition_name, leader = CONDITIONS.get(condition, ("", ""))

        print(f"[{i}/{len(takes)}] {duo}/{take}  ({stem}; "
              + ", ".join(f"{p} {r}" for p, r in sorted(roles.items()))
              + f"; role margin {margin:.2f})")

        for player, midi_fn in sorted(files.items()):
            role = roles[player]
            name = performance_name(stem, role, duo, take, player)
            paths = performance_paths(args.layout, stem, role, duo, take, player)
            score_rel = f"{prefix}/{score_path(stem, role)}"
            midi_out = os.path.join(out, paths["midi"])
            match_out = os.path.join(out, paths["match"])
            csv_out = os.path.join(out, paths["parangonada"])
            os.makedirs(os.path.dirname(midi_out), exist_ok=True)

            if not os.path.exists(midi_out) or args.overwrite:
                shutil.copyfile(midi_fn, midi_out)

            ppart = performances[player]
            spart = piece.parts[role]
            if os.path.exists(match_out) and not args.overwrite:
                alignment = pt.load_match(match_out, create_score=False)[1]
            else:
                alignment = matcher(piece.note_arrays[role], ppart.note_array())
                pt.save_match(
                    alignment, ppart, spart, out=match_out,
                    piece=f"{title} ({role})",
                    performer=f"{duo} {player}",
                    score_filename=os.path.basename(score_rel),
                    performance_filename=os.path.basename(midi_out),
                )
                os.makedirs(csv_out, exist_ok=True)
                pt.save_parangonada_csv(alignment, ppart, spart, outdir=csv_out)

            labels = np.array([a["label"] for a in alignment])
            counts = {
                "matched": int(np.sum(labels == "match")),
                "inserted": int(np.sum(labels == "insertion")),
                "deleted": int(np.sum(labels == "deletion")),
            }
            print(f"    {player} {role:<8s} {name}: {counts['matched']} matched, "
                  f"{counts['inserted']} inserted, {counts['deleted']} deleted")

            tempo_rel = os.path.join(out, paths["tempo_curves"])
            avatar_rel = os.path.join(out, paths["avatar"])
            motion_rel = os.path.join(out, paths["motion"], "motion.npz")
            rows.append({
                "score": score_rel,
                "midi": f"{prefix}/{paths['midi']}",
                "match": f"{prefix}/{paths['match']}",
                "parangonada": f"{prefix}/{paths['parangonada']}",
                "tempo_curves": f"{prefix}/{paths['tempo_curves']}" if os.path.exists(tempo_rel) else "",
                "avatar": f"{prefix}/{paths['avatar']}" if os.path.exists(avatar_rel) else "",
                "motion": f"{prefix}/{paths['motion']}" if os.path.exists(motion_rel) else "",
                "piece": stem,
                "title": title,
                "part": role,
                "duo": duo,
                "take": take,
                "block": block,
                "trial": trial,
                "condition": condition,
                "condition_name": condition_name,
                "leader": leader,
                "player": player,
                "performed_notes": int(len(ppart.notes)),
                "score_notes": int(len(piece.note_arrays[role])),
                **counts,
                "source_midi": os.path.abspath(midi_fn),
            })

    write_metadata(out, rows, list(METADATA_COLUMNS))

    built = sorted({(row["piece"], row["title"], row["block"]) for row in rows})
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(README.format(
            layout_section=LAYOUT_SECTIONS[args.layout],
            pieces_line=", ".join(f"`{stem}` (*{title}*)" for stem, title, _ in built),
            blocks_line="; ".join(f"block `{block}` is the {stem}" for stem, _, block in built),
        ))

    print(f"\n{len(rows)} performances in {time.perf_counter() - started:.0f}s")
    print(f"metadata: {os.path.join(out, 'metadata-aura.csv')}")


if __name__ == "__main__":
    main()
