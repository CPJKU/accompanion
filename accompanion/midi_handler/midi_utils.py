# -*- coding: utf-8 -*-
"""
MIDI utilities

Writing a performance out is what these do. The times handed in are seconds on
one clock -- the ACCompanion's own performance clock, which the sequencer and
the MIDI input already share -- and they stay on it: a soloist track and an
accompaniment track written from the same session line up with each other,
because neither is rebased to its own first note.
"""
import os
from os.path import dirname, realpath

import mido

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
acc_pack_dir = dirname(dir_of_file)
accdir = dirname(acc_pack_dir)


#: Where a recording goes when the caller names no directory. Created when
#: something is actually written, not on import.
OUTPUT_MIDI_FOLDER = os.path.join(accdir, "recorded_midi")

#: Channel messages worth keeping. Clock, active sensing and sysex traffic
#: would grow the file without describing anything that was played. Control
#: changes are kept: the sustain pedal is part of the performance.
RECORDED_TYPES = frozenset(
    {
        "note_on",
        "note_off",
        "control_change",
        "program_change",
        "pitchwheel",
        "aftertouch",
        "polytouch",
    }
)

TICKS_PER_BEAT = 480
#: Recorded times are wall-clock seconds, not score beats, so a recording
#: carries one fixed tempo and its tick grid stays linear in seconds.
MICROSECONDS_PER_BEAT = 500000


def write_midi(
    output_path,
    tracks,
    ticks_per_beat=TICKS_PER_BEAT,
    tempo=MICROSECONDS_PER_BEAT,
):
    """Write named tracks of ``(message, seconds)`` items as one MIDI file.

    Every track shares the caller's timeline, so several written from one
    session can be laid over each other without alignment. Absolute seconds
    are rounded to absolute ticks before differencing, so rounding cannot
    accumulate over a long performance the way per-delta rounding does.

    Args:
        output_path str: where to write the file
        tracks list: ``(track name, [(mido.Message, seconds), ...])`` pairs.
            An empty track is written as an empty track, not skipped.
        ticks_per_beat int: the file's time division
        tempo int: microseconds per beat, written as a `set_tempo` meta
            message so the file's seconds survive being read back
    """
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    for name, messages in tracks:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
        previous = 0
        for message, seconds in sorted(messages, key=lambda item: item[1]):
            ticks = round(mido.second2tick(max(0.0, seconds), ticks_per_beat, tempo))
            track.append(message.copy(time=ticks - previous))
            previous = ticks
        midi.tracks.append(track)
    midi.save(output_path)
    return output_path


def close_sounding_notes(messages, end):
    """Append a note off for every note the messages leave sounding.

    A recording taken at a port does not see the all-notes-off that silences
    the instrument on shutdown -- the panic button goes straight to the
    hardware -- so without this a held chord is never released and the file
    reads as one stuck sound.

    Args:
        messages list: ``(mido.Message, seconds)`` pairs
        end float: when to release, in the same seconds
    """
    sounding = []
    for message, _ in messages:
        key = (getattr(message, "channel", 0), getattr(message, "note", None))
        if key[1] is None:
            continue
        if message.type == "note_on" and message.velocity:
            if key not in sounding:
                sounding.append(key)
        elif message.type in ("note_off", "note_on") and key in sounding:
            sounding.remove(key)
    return list(messages) + [
        (mido.Message("note_off", channel=channel, note=note, velocity=0), end)
        for channel, note in sounding
    ]


def first_note_time(messages):
    """When the first note sounds, in the caller's seconds, or None."""
    return min(
        (
            seconds
            for message, seconds in messages
            if message.type == "note_on" and message.velocity
        ),
        default=None,
    )


def midi_file_from_midi_msg(midi_msg_list, output_path):
    """Save a midi file, given a sequence of midi messages with a absolute time stamp.

    Kept for callers that write a single part on its own. The file starts at
    the caller's time zero, so two files written this way still share a
    timeline; `write_midi` is the one to use for several parts at once.

    Args:
        midi_msg_list list: the list of pairs (midi_msg, time_stamp)
        output_path str: the path of the output midi file
    """
    return write_midi(output_path, [("performance", list(midi_msg_list))])
