#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The two layouts of the ``aura-data`` folder, and moving between them.

``nested`` -- one folder per take, as the recordings are delivered::

    aura-data/
        README.md
        metadata-aura.csv
        score/Badinerie_primo_xml_score.musicxml
        <duo>/<take>/midi_p1.mid                 the performance
        <duo>/<take>/midi_p1.match               its alignment
        <duo>/<take>/parangonada_p1/             the alignment for parangonada
        <duo>/<take>/tempo_curves_p1.npz         bin/extract_tempo_curves.py
        <duo>/<take>/tempo_curves.json
        <duo>/<take>/avatar_p2.npz               bin/build_avatar_features.py
        <duo>/<take>/video_p1_top/motion.npz     bin/add_aura_motion.py (pyanoduo's
                                                 prepared MHR motion, plus its
                                                 summary, metrics and diagnostics;
                                                 named as pyanoduo's own cache is)
        <duo>/<take>/sync.csv                    bin/write_aura_sync.py
        tempo_curves-index.json, avatar-index.json

``flat`` -- one folder per kind of file, as in ``data/asap``::

    aura-data/
        README.md
        metadata-aura.csv
        score/Badinerie_primo_xml_score.musicxml
        midi/<name>.mid
        match/<name>.match
        parangonada/<name>/
        tempo_curves/<duo>/<take>/<player>.npz, summary.json, index.json
        avatar/<name>.npz, index.json
        motion/<name>/motion.npz, ...

with ``<name>`` = ``<Piece>_<part>_<duo>_<take>_<player>``.

In both, ``metadata-aura.csv`` is the source of truth: one row per
performance saying which piece and part is in which file. In the nested
layout the file names alone do not say which part a player played -- the
players swap parts between takes -- so read the ``part`` column.

Usage
-----
    python bin/aura_layout.py --data ~/datasets/aura-data --to nested
    python bin/aura_layout.py --data ~/datasets/aura-data --to flat

Converting is a pure rename: nothing is rewritten but the paths in the
metadata and the layout section of the README. Anything written in the
other layout since -- a fresh ``tempo_curves/`` tree from
`extract_tempo_curves.py`, say -- is folded in too, so ``--to`` can be run
again at any time to tidy up.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys

LAYOUTS = ("nested", "flat")

#: Per-performance collections besides the alignment: ``<folder>`` at the
#: top level in the flat layout, ``<folder>_<player>.npz`` per take in the
#: nested one. Matched by prefix so that variants (``avatar-offline-self``)
#: come along.
COLLECTIONS = ("avatar",)

PATH_COLUMNS = ("score", "midi", "match", "parangonada", "tempo_curves", "avatar", "motion")


def performance_name(piece, part, duo, take, player):
    return f"{piece}_{part}_{duo}_{take}_{player}"


def score_path(piece, part):
    """Relative to the data folder; the same in both layouts."""
    return f"score/{piece}_{part}_xml_score.musicxml"


def performance_paths(layout, piece, part, duo, take, player):
    """Where one performance's files live, relative to the data folder."""
    name = performance_name(piece, part, duo, take, player)
    if layout == "flat":
        return {
            "midi": f"midi/{name}.mid",
            "match": f"match/{name}.match",
            "parangonada": f"parangonada/{name}",
            "tempo_curves": f"tempo_curves/{duo}/{take}/{player}.npz",
            "tempo_summary": f"tempo_curves/{duo}/{take}/summary.json",
            "avatar": f"avatar/{name}.npz",
            "motion": f"motion/{name}",
        }
    if layout == "nested":
        base = f"{duo}/{take}"
        return {
            "midi": f"{base}/midi_{player}.mid",
            "match": f"{base}/midi_{player}.match",
            "parangonada": f"{base}/parangonada_{player}",
            "tempo_curves": f"{base}/tempo_curves_{player}.npz",
            "tempo_summary": f"{base}/tempo_curves.json",
            "avatar": f"{base}/avatar_{player}.npz",
            "motion": f"{base}/video_{player}_top",
        }
    raise ValueError(f"Unknown layout '{layout}'")


def collection_path(layout, collection, piece, part, duo, take, player):
    """One collection's file for one performance (``avatar`` and variants)."""
    if layout == "flat":
        return f"{collection}/{performance_name(piece, part, duo, take, player)}.npz"
    return f"{duo}/{take}/{collection}_{player}.npz"


def index_path(layout, collection):
    """The collection's index.json."""
    return f"{collection}/index.json" if layout == "flat" else f"{collection}-index.json"


def read_metadata(data):
    with open(os.path.join(data, "metadata-aura.csv"), newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames)


def write_metadata(data, rows, columns):
    for column in PATH_COLUMNS:
        if column not in columns:
            columns.append(column)
    rows.sort(key=lambda r: (r["piece"], r["part"], r["duo"], r["take"], r["player"]))
    with open(os.path.join(data, "metadata-aura.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def detect_layout(data):
    """Which layout the folder is in, read off the metadata's midi paths."""
    rows, _ = read_metadata(data)
    if not rows:
        return "flat"
    midi = rows[0]["midi"].split("/", 1)[-1]
    return "flat" if midi.startswith("midi/") else "nested"


def _move(src, dst, moved):
    if not os.path.exists(src) or os.path.abspath(src) == os.path.abspath(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        raise FileExistsError(f"Refusing to overwrite {dst} with {src}")
    shutil.move(src, dst)
    moved.append((src, dst))


def _prune(data):
    """Remove the folders the move left empty."""
    for root, _, _ in os.walk(data, topdown=False):
        if root != data and not os.listdir(root):
            os.rmdir(root)


def collections_present(data, layout, rows):
    """Names of the avatar-like collections that have files in `data`."""
    found = set()
    for prefix in COLLECTIONS:
        if layout == "flat":
            for path in glob.glob(os.path.join(data, f"{prefix}*")):
                if os.path.isdir(path):
                    found.add(os.path.basename(path))
        else:
            for path in glob.glob(os.path.join(data, "*", "*", f"{prefix}*_p?.npz")):
                found.add(re.sub(r"_p\d\.npz$", "", os.path.basename(path)))
            for path in glob.glob(os.path.join(data, f"{prefix}*-index.json")):
                found.add(os.path.basename(path)[: -len("-index.json")])
    return sorted(found)


def convert(data, target, verbose=True):
    """Bring `data` into the `target` layout. Returns the number of moves."""
    data = os.path.abspath(data)
    prefix = os.path.basename(data)
    rows, columns = read_metadata(data)
    source = "flat" if target == "nested" else "nested"
    moved = []

    # Any collection present in either layout is folded into the target.
    collections = set(collections_present(data, source, rows)) | set(
        collections_present(data, target, rows)
    )

    for row in rows:
        key = (row["piece"], row["part"], row["duo"], row["take"], row["player"])
        src = performance_paths(source, *key)
        dst = performance_paths(target, *key)
        for kind in ("midi", "match", "parangonada", "tempo_curves", "tempo_summary", "motion"):
            _move(os.path.join(data, src[kind]), os.path.join(data, dst[kind]), moved)
        for collection in collections:
            _move(
                os.path.join(data, collection_path(source, collection, *key)),
                os.path.join(data, collection_path(target, collection, *key)),
                moved,
            )
        row["score"] = f"{prefix}/{score_path(row['piece'], row['part'])}"
        for kind in ("midi", "match", "parangonada", "tempo_curves", "motion"):
            row[kind] = (
                f"{prefix}/{dst[kind]}" if os.path.exists(os.path.join(data, dst[kind])) else ""
            )
        avatar = collection_path(target, "avatar", *key)
        row["avatar"] = f"{prefix}/{avatar}" if os.path.exists(os.path.join(data, avatar)) else ""

    for collection in sorted(collections | {"tempo_curves"}):
        _move(
            os.path.join(data, index_path(source, collection)),
            os.path.join(data, index_path(target, collection)),
            moved,
        )

    _prune(data)
    write_metadata(data, rows, columns)
    _update_other_metadata(data, rows)
    update_readme(data, target)
    if verbose:
        print(f"{len(moved)} file(s) moved; {data} is now in the {target} layout")
    return len(moved)


def _update_other_metadata(data, rows):
    """Carry the new paths into any other ``metadata-aura*.csv`` in `data`.

    A filtered copy of the metadata -- the secondo performances only, say --
    keeps working after a conversion: its path columns are rewritten from
    the main metadata, row by row, and its other columns are left alone.
    """
    key = lambda r: (r["piece"], r["part"], r["duo"], r["take"], r["player"])  # noqa: E731
    current = {key(r): r for r in rows}
    for fn in glob.glob(os.path.join(data, "metadata-aura*.csv")):
        if os.path.basename(fn) == "metadata-aura.csv":
            continue
        with open(fn, newline="") as f:
            reader = csv.DictReader(f)
            other_rows, columns = list(reader), list(reader.fieldnames)
        for row in other_rows:
            match = current.get(key(row))
            if match is None:
                continue
            for column in PATH_COLUMNS:
                if column in columns:
                    row[column] = match[column]
        with open(fn, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(other_rows)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
LAYOUT_SECTIONS = {
    "nested": """\
## Layout

One folder per duo and take, as the recordings are delivered:

    metadata-aura.csv                       one row per performance
    score/<Piece>_<part>_xml_score.musicxml the part's score
    <duo>/<take>/midi_p1.mid                the performance, byte for byte as recorded
    <duo>/<take>/midi_p1.match              its alignment (partitura match file, v1.1.0)
    <duo>/<take>/parangonada_p1/            the same alignment as parangonada CSVs
    <duo>/<take>/tempo_curves_p1.npz        tempo curves (bin/extract_tempo_curves.py)
    <duo>/<take>/tempo_curves.json          the take's online-vs-offline summary
    <duo>/<take>/avatar_p2.npz              avatar input streams and features
                                            (bin/build_avatar_features.py)
    <duo>/<take>/video_p1_top/motion.npz    the player's motion, as pyanoduo prepares it
                                            (MHR parameters per video frame), with its
                                            summary.json, metrics.csv, diagnostics.png
    <duo>/<take>/sync.csv                   file offsets on the shared timeline, as
                                            pyanoduo's training reads them
    tempo_curves-index.json, avatar-index.json

`p1` / `p2` is the file the note came from, not the part: read the `part`
column of the metadata to know which part a file holds (see below).
""",
    "flat": """\
## Layout

One folder per kind of file, like the `asap` folder matchmaker's benchmark uses:

    metadata-aura.csv                       one row per performance
    score/<Piece>_<part>_xml_score.musicxml the part's score
    midi/<name>.mid                         the performance, byte for byte as recorded
    match/<name>.match                      its alignment (partitura match file, v1.1.0)
    parangonada/<name>/                     the same alignment as parangonada CSVs
    tempo_curves/<duo>/<take>/<player>.npz  tempo curves (bin/extract_tempo_curves.py)
    avatar/<name>.npz                       avatar input streams and features
                                            (bin/build_avatar_features.py)
    motion/<name>/motion.npz                the player's motion, as pyanoduo prepares it

with `<name>` = `<Piece>_<part>_<duo>_<take>_<player>`, e.g.
`Badinerie_primo_D05_B1_T1_L1_p1`.
""",
}


def update_readme(data, layout):
    """Swap the README's layout section for the one describing `layout`."""
    fn = os.path.join(data, "README.md")
    if not os.path.exists(fn):
        return
    text = open(fn).read()
    pattern = re.compile(r"## Layout\n.*?(?=\n## )", re.S)
    if pattern.search(text):
        text = pattern.sub(LAYOUT_SECTIONS[layout].rstrip("\n"), text, count=1)
        open(fn, "w").write(text)


def main():
    parser = argparse.ArgumentParser(
        description="Convert the aura-data folder between its two layouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", required=True, help="the aura-data folder")
    parser.add_argument("--to", choices=LAYOUTS, help="target layout")
    parser.add_argument("--status", action="store_true", help="only report the layout")
    args = parser.parse_args()
    if args.status or not args.to:
        print(f"{os.path.abspath(args.data)}: {detect_layout(args.data)} layout")
        return
    convert(args.data, args.to)


if __name__ == "__main__":
    main()
