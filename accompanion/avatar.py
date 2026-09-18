# -*- coding: utf-8 -*-
"""
What the avatar gets to see of the ACCompanion's own playing.

The ACCompanion plays its part through `ScoreSequencer`: a `note_on` at the
scheduled time of every accompaniment note and a `note_off` at its end, each
backed by a score note whose onset and duration in beats it knows. Next to
that, the following loop keeps a dead-reckoned score position and a beat
period on every input frame, and knows when it is holding at a fermata. That
is everything the machine has about its own performance as it happens, and
everything an avatar of its body can be driven from.

This module fixes that information as two event streams, and turns them into
a feature vector per motion frame::

    NoteEvent   one per MIDI message the ACCompanion sends for its part:
                time_sec, on, pitch, velocity, score_onset_beat,
                score_duration_beat, note_id
    FrameEvent  one per following frame (every polling period):
                time_sec, position_beat, beat_period, waiting

Both are stamped with the ACCompanion's performance clock, seconds since it
started listening (`follow_step`'s `solo_p_onset`, the sequencer's `c_time`).

The same streams can be written from a recorded human performance of the
part -- the MIDI file gives the notes, a match file gives each note its score
note, and a tempo curve gives the frames -- which is how training data is
made: `bin/build_avatar_features.py` does that for the duo dataset. Because
`AvatarFeaturizer` only ever sees the streams, the features it computes from
a recording and the ones it computes live are the same function of the same
kind of input; the one liberty the offline builder takes is choosing *which*
position estimate stands in for the live one (see there).

Features, per motion frame at time ``t``, all causal in the streams and in
physical units (see `AvatarFeaturizer.feature_names`):

``key_down[88]``
    Velocity / 127 of every key the part is holding down, 0 otherwise.
``key_recency[88]``
    ``exp(-(t - last onset on that key) / recency_tau)``: where the hands were
    a moment ago, with sub-frame timing.
``since_onset_sec``, ``to_next_onset_sec``
    Seconds since the part's last onset, and until its next one as predicted
    from the score at the current tempo. Both clipped at ``horizon_sec``.
``position_beat``, ``beat_phase``, ``bar_phase``, ``progress``
    The dead-reckoned score position in beats; its fraction within the beat
    and within the bar; and the share of the piece behind us.
``beat_period_sec``, ``waiting``
    The tempo, and 1 while the ACCompanion is holding at a fermata.
``lookahead[K x 5]``
    The next ``K`` score notes of the part not played yet, each as
    ``(valid, pitch, delta_beat, delta_sec, duration_beat)`` -- what a
    pianist's hands are already moving towards. ``delta_sec`` is
    ``delta_beat * beat_period``. Slots past the end of the piece have
    ``valid`` 0 and zeros elsewhere.

Notes the score does not explain
--------------------------------
A recorded human plays notes that are not in the score, and leaves out notes
that are. The ACCompanion does neither. Both cases are handled so that the
training input keeps the two invariants the live input has:

*A key press is a key press.* A performed note the alignment could not place
still goes into the stream, with no score note (``score_onset_beat`` NaN,
``note_id`` empty): the hands moved for it, and the motion shows that. It
counts for ``key_down``, ``key_recency`` and ``since_onset``; it does not
mark any score note as played.

*Every note the lookahead announces gets played.* Live that holds by
construction: the sequencer plays every note of the part. In a recording,
the score notes the player left out are passed as ``skipped_ids`` and never
enter the lookahead, so the policy is not shown hands moving towards a note
that then does not come. (A note played more than a beat after the position
passed its onset is treated the same way: dropped from the lookahead, and
then a plain key press when it does come.)

Score *position* is never a problem either way: it is a property of time,
carried by the frame stream, not of notes. Every motion frame has a beat
position whether or not a note sounding at that moment has a score note.

Nothing here imports beyond numpy, so the module can be used from the motion
side (pyanoduo) as it is.
"""
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: The 88 keys of the piano, MIDI pitches 21..108.
LOWEST_PITCH = 21
N_KEYS = 88

NOTE_EVENT_DTYPE = np.dtype([
    ("time_sec", "f8"),
    ("on", "?"),
    ("pitch", "i2"),
    ("velocity", "i2"),
    #: NaN for a note the score does not have (never the case live).
    ("score_onset_beat", "f8"),
    ("score_duration_beat", "f8"),
    ("note_id", "U16"),
])

FRAME_EVENT_DTYPE = np.dtype([
    ("time_sec", "f8"),
    ("position_beat", "f8"),
    ("beat_period", "f8"),
    ("waiting", "?"),
])

SCORE_NOTE_DTYPE = np.dtype([
    ("note_id", "U16"),
    ("pitch", "i2"),
    ("onset_beat", "f8"),
    ("duration_beat", "f8"),
])


# ---------------------------------------------------------------------------
# Building the streams
# ---------------------------------------------------------------------------
def note_event(time_sec, on, pitch, velocity, score_onset_beat=np.nan,
               score_duration_beat=np.nan, note_id=""):
    """One `NoteEvent` row."""
    return np.array(
        [(time_sec, on, pitch, velocity, score_onset_beat, score_duration_beat, note_id)],
        dtype=NOTE_EVENT_DTYPE,
    )[0]


def frame_event(time_sec, position_beat, beat_period, waiting=False):
    """One `FrameEvent` row."""
    return np.array(
        [(time_sec, position_beat, beat_period, waiting)], dtype=FRAME_EVENT_DTYPE
    )[0]


def note_event_from_sequencer(note, on, time_sec):
    """The `NoteEvent` for a message `ScoreSequencer` is about to send.

    `note` is an `accompanion.accompanist.score.Note`; call this right where
    the sequencer does ``self.outport.send(n_on.note_on)`` (``on=True``) and
    ``self.outport.send(n_off.note_off)`` (``on=False``), with the
    sequencer's ``c_time``.
    """
    return note_event(
        time_sec=time_sec,
        on=on,
        pitch=int(note.pitch),
        velocity=int(note.velocity),
        score_onset_beat=float(note.onset),
        score_duration_beat=float(note.duration),
        note_id=str(note.id),
    )


def frame_event_from_following(time_sec, state, tempo_model, fermata_hold):
    """The `FrameEvent` at the end of one `ACCompanion.follow_step`.

    `state` is the `FollowingState`, `tempo_model` the ACCompanion's, and
    `fermata_hold` its `FermataHold`; `time_sec` the frame's `solo_p_onset`.
    """
    return frame_event(
        time_sec=time_sec,
        position_beat=float(state.expected_position),
        beat_period=float(tempo_model.beat_period),
        waiting=bool(fermata_hold.waiting),
    )


def score_notes_from_accompaniment(notes) -> np.ndarray:
    """`SCORE_NOTE_DTYPE` rows from `AccompanimentScore.notes`."""
    rows = [(str(n.id), int(n.pitch), float(n.onset), float(n.duration)) for n in notes]
    return np.array(rows, dtype=SCORE_NOTE_DTYPE)


def score_notes_from_note_array(note_array) -> np.ndarray:
    """`SCORE_NOTE_DTYPE` rows from a partitura score note array."""
    rows = [
        (str(n["id"]), int(n["pitch"]), float(n["onset_beat"]), float(n["duration_beat"]))
        for n in note_array
    ]
    return np.array(rows, dtype=SCORE_NOTE_DTYPE)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
class AvatarFeaturizer(object):
    """Turns the two streams into one feature vector per motion frame.

    Feed it events in time order with `push_note` / `push_frame`, and ask
    for `features(t)` at any ``t`` at or after the last event it was given.
    Or hand it whole streams and a vector of timestamps with `run`.

    Parameters
    ----------
    score_notes : np.ndarray (SCORE_NOTE_DTYPE)
        The part the ACCompanion plays, for the lookahead.
    lookahead : int
        How many upcoming score notes to describe.
    recency_tau : float
        Time constant of ``key_recency``, in seconds.
    horizon_sec : float
        Where the two onset-distance features are clipped, in seconds.
    beats_per_bar : float
        Length of a bar in beats, for ``bar_phase``. The badinerie is in 4/4.
    skipped_ids : iterable of str
        Ids of score notes this performance never plays. Left out of the
        lookahead; see the module docstring. Empty live.
    """

    def __init__(
        self,
        score_notes: np.ndarray,
        lookahead: int = 8,
        recency_tau: float = 0.25,
        horizon_sec: float = 5.0,
        beats_per_bar: float = 4.0,
        skipped_ids: Iterable[str] = (),
    ) -> None:
        self.score_notes = np.sort(
            np.asarray(score_notes, dtype=SCORE_NOTE_DTYPE), order=["onset_beat", "pitch"]
        )
        self.lookahead = int(lookahead)
        self.recency_tau = float(recency_tau)
        self.horizon_sec = float(horizon_sec)
        self.beats_per_bar = float(beats_per_bar)
        ends = self.score_notes["onset_beat"] + self.score_notes["duration_beat"]
        self.first_beat = float(self.score_notes["onset_beat"].min()) if len(ends) else 0.0
        self.total_beats = float(ends.max() - self.first_beat) if len(ends) else 1.0
        self._score_index = {nid: i for i, nid in enumerate(self.score_notes["note_id"])}
        self.skipped = np.isin(self.score_notes["note_id"], list(skipped_ids))
        self.reset()

    def reset(self) -> None:
        self.key_velocity = np.zeros(N_KEYS)
        self.key_last_onset = np.full(N_KEYS, -np.inf)
        self.last_onset = -np.inf
        self.played = np.zeros(len(self.score_notes), dtype=bool)
        self.frame: Optional[np.void] = None
        self.last_time = -np.inf

    # -- input ---------------------------------------------------------------
    def push_note(self, event) -> None:
        key = int(event["pitch"]) - LOWEST_PITCH
        t = float(event["time_sec"])
        self.last_time = max(self.last_time, t)
        if not 0 <= key < N_KEYS:
            return
        if event["on"] and event["velocity"] > 0:
            self.key_velocity[key] = float(event["velocity"])
            self.key_last_onset[key] = t
            self.last_onset = t
            index = self._score_index.get(str(event["note_id"]))
            if index is not None:
                self.played[index] = True
        else:
            self.key_velocity[key] = 0.0

    def push_frame(self, event) -> None:
        self.frame = event
        self.last_time = max(self.last_time, float(event["time_sec"]))

    # -- output --------------------------------------------------------------
    @property
    def dim(self) -> int:
        return 2 * N_KEYS + 2 + 6 + 5 * self.lookahead

    @property
    def feature_names(self) -> List[str]:
        names = [f"key_down_{p}" for p in range(LOWEST_PITCH, LOWEST_PITCH + N_KEYS)]
        names += [f"key_recency_{p}" for p in range(LOWEST_PITCH, LOWEST_PITCH + N_KEYS)]
        names += ["since_onset_sec", "to_next_onset_sec"]
        names += ["position_beat", "beat_phase", "bar_phase", "progress",
                  "beat_period_sec", "waiting"]
        for k in range(self.lookahead):
            names += [f"lookahead_{k}_{c}" for c in
                      ("valid", "pitch", "delta_beat", "delta_sec", "duration_beat")]
        return names

    def position_at(self, t: float) -> Tuple[float, float, bool]:
        """``(position, beat_period, waiting)`` at `t`.

        The position is carried forward from the last frame at the tempo,
        as the following loop itself does between frames, and stands still
        while waiting.
        """
        if self.frame is None:
            return self.first_beat, np.nan, False
        position = float(self.frame["position_beat"])
        beat_period = float(self.frame["beat_period"])
        waiting = bool(self.frame["waiting"])
        elapsed = t - float(self.frame["time_sec"])
        if not waiting and elapsed > 0 and beat_period > 0:
            position += elapsed / beat_period
        return position, beat_period, waiting

    def upcoming(self, position: float) -> np.ndarray:
        """The next `lookahead` unplayed score notes from `position` on.

        A note left unplayed more than a beat behind the position is taken
        as skipped rather than as still coming.
        """
        pending = (
            ~self.played & ~self.skipped
            & (self.score_notes["onset_beat"] >= position - 1.0)
        )
        return self.score_notes[pending][: self.lookahead]

    def features(self, t: float) -> np.ndarray:
        out = np.zeros(self.dim)
        position, beat_period, waiting = self.position_at(t)
        bp = beat_period if np.isfinite(beat_period) and beat_period > 0 else 0.0

        out[:N_KEYS] = self.key_velocity / 127.0
        with np.errstate(over="ignore", invalid="ignore"):
            age = t - self.key_last_onset
            recency = np.where(np.isfinite(age), np.exp(-age / self.recency_tau), 0.0)
        out[N_KEYS:2 * N_KEYS] = recency

        i = 2 * N_KEYS
        since = t - self.last_onset if np.isfinite(self.last_onset) else self.horizon_sec
        out[i] = min(max(since, 0.0), self.horizon_sec)

        ahead = self.upcoming(position)
        if len(ahead) and bp > 0:
            to_next = max(float(ahead["onset_beat"][0]) - position, 0.0) * bp
        else:
            to_next = self.horizon_sec
        out[i + 1] = min(to_next, self.horizon_sec)

        i += 2
        rel = position - self.first_beat
        out[i:i + 6] = [
            position,
            rel - np.floor(rel),
            (rel % self.beats_per_bar) / self.beats_per_bar,
            min(max(rel / self.total_beats, 0.0), 1.0),
            bp,
            1.0 if waiting else 0.0,
        ]

        i += 6
        for k, note in enumerate(ahead):
            delta = max(float(note["onset_beat"]) - position, 0.0)
            out[i + 5 * k:i + 5 * k + 5] = [
                1.0, float(note["pitch"]), delta, delta * bp, float(note["duration_beat"])
            ]
        return out

    def run(self, note_events: np.ndarray, frame_events: np.ndarray,
            timestamps: Sequence[float]) -> np.ndarray:
        """Features at each of `timestamps` (ascending), from whole streams.

        Every event at or before a timestamp is known at it; nothing later
        is. Returns ``(len(timestamps), dim)``.
        """
        self.reset()
        notes = np.sort(np.asarray(note_events, dtype=NOTE_EVENT_DTYPE), order="time_sec")
        frames = np.sort(np.asarray(frame_events, dtype=FRAME_EVENT_DTYPE), order="time_sec")
        timestamps = np.asarray(timestamps, dtype=float)
        out = np.zeros((len(timestamps), self.dim))
        ni = fi = 0
        for row, t in enumerate(timestamps):
            # Frames first, then notes, so that a note stamped like a frame
            # is read against that frame's position.
            while fi < len(frames) and frames["time_sec"][fi] <= t:
                self.push_frame(frames[fi])
                fi += 1
            while ni < len(notes) and notes["time_sec"][ni] <= t:
                self.push_note(notes[ni])
                ni += 1
            out[row] = self.features(t)
        return out


# ---------------------------------------------------------------------------
# Feeding pyanoduo
# ---------------------------------------------------------------------------
class AvatarConditioner(object):
    """The avatar streams of ``aura-data`` as a pyanoduo `MidiConditioner`.

    pyanoduo asks a conditioner for ``feature_dim`` and for
    ``features(metadata, timestamps)`` -- one row per motion frame, on the
    motion's own clock. This answers from the ``avatar_<player>.npz`` files
    `bin/build_avatar_features.py` writes: the streams are loaded for the
    recording `metadata` names and run through `AvatarFeaturizer` at the
    requested timestamps, so a training run and a later live run see the
    same function of the same kind of input.

    Duck-typed rather than subclassing pyanoduo's ABC, so that this module
    stays free of any import beyond numpy.

    Parameters
    ----------
    data : str
        The ``aura-data`` folder, in either layout.
    collection : str
        ``"avatar"`` (position from the ACCompanion following the primo) or
        ``"avatar-offline-self"`` (ground-truth position).
    """

    def __init__(self, data: str, collection: str = "avatar") -> None:
        import json
        import os

        self.data = os.path.abspath(data)
        self.collection = collection
        for index in (f"{collection}-index.json", os.path.join(collection, "index.json")):
            path = os.path.join(self.data, index)
            if os.path.exists(path):
                with open(path) as f:
                    self.index = json.load(f)
                break
        else:
            raise FileNotFoundError(f"No index for '{collection}' under {self.data}")
        self._cache = {}

    @property
    def feature_dim(self) -> int:
        return int(self.index["feature_dim"])

    @property
    def feature_names(self) -> List[str]:
        return list(self.index["feature_names"])

    def file_for(self, metadata) -> str:
        """The npz of the recording `metadata` (duo, recording, participant) names."""
        import glob
        import os

        duo, take, player = metadata.duo, metadata.recording, metadata.participant
        nested = os.path.join(self.data, duo, take, f"{self.collection}_{player}.npz")
        if os.path.exists(nested):
            return nested
        flat = glob.glob(os.path.join(self.data, self.collection, f"*_{duo}_{take}_{player}.npz"))
        if flat:
            return flat[0]
        raise FileNotFoundError(
            f"No '{self.collection}' file for {duo}/{take} {player} under {self.data}: "
            "the avatar streams exist only for the player of the part the avatar "
            "plays (the secondo), and pyanoduo's target is always p2."
        )

    def load(self, metadata):
        """``(npz, featurizer)`` for a recording, cached."""
        path = self.file_for(metadata)
        if path not in self._cache:
            d = np.load(path)
            featurizer = AvatarFeaturizer(
                d["score_notes"],
                lookahead=int(d["lookahead"]),
                recency_tau=float(d["recency_tau"]),
                horizon_sec=float(d["horizon_sec"]),
                beats_per_bar=float(d["beats_per_bar"]),
                skipped_ids=d["skipped_note_ids"],
            )
            self._cache[path] = (d, featurizer)
        return self._cache[path]

    def features(self, metadata, timestamps) -> np.ndarray:
        d, featurizer = self.load(metadata)
        out = featurizer.run(d["note_events"], d["frame_events"], timestamps)
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# pyanoduo's MIDI-conditioned pipeline (branch dataset-midi-train)
# ---------------------------------------------------------------------------
# That pipeline encodes the music itself (a piano roll on a beat grid) and
# keeps two boundaries open: an `AccompanimentSource` whose `notes_at(time)`
# returns the secondo's currently scheduled notes, and a `ScoreClock` that
# takes `(time, beat, beat_period)` observations from anywhere. These two
# adapters hand it the ACCompanion's own schedule and clock, so that at
# inference the avatar runs on the same follower that plays the accompaniment
# rather than on a second follower of its own.

SCHEDULE_DTYPE = np.dtype([
    ("id", "U16"),
    ("pitch", "i4"),
    ("onset_sec", "f4"),
    ("duration_sec", "f4"),
    ("velocity", "i4"),
])


class AccompanionSchedule(object):
    """The ACCompanion's live schedule of its part, as an `AccompanimentSource`.

    `notes` is `AccompanimentScore.notes`: every note of the part with the
    performed onset (`p_onset`), duration (`p_duration`) and velocity the
    accompanist has currently assigned it. `accompaniment_step` keeps
    revising the future ones on every solo onset; the ones already sent are
    fixed. `notes_at` returns the whole schedule as it stands, on the
    ACCompanion's performance clock plus `offset` (the schedule's zero on the
    shared timeline, as `sync.csv` gives it offline).
    """

    def __init__(self, notes, offset: float = 0.0) -> None:
        self.notes = list(notes)
        self.offset = float(offset)

    def notes_at(self, time: float) -> np.ndarray:
        rows = []
        for note in self.notes:
            onset = note.p_onset
            if onset is None:
                continue
            duration = note.p_duration if note.p_duration is not None else 0.0
            rows.append((str(note.id), int(note.pitch), float(onset) + self.offset,
                         float(duration), int(note.velocity)))
        return np.array(rows, dtype=SCHEDULE_DTYPE)


def clock_samples(frame_events: np.ndarray, times: Sequence[float], offset: float = 0.0):
    """``(times, beats, periods)`` of the ACCompanion's clock at `times`.

    The shape of pyanoduo's `ClockTrace`: for training on the ACCompanion's
    clock rather than pyanoduo's own follower. The position is carried
    between frames as the following loop does (see
    `AvatarFeaturizer.position_at`), standing still while waiting.
    """
    frames = np.sort(np.asarray(frame_events, dtype=FRAME_EVENT_DTYPE), order="time_sec")
    times = np.asarray(times, dtype=float)
    ft = frames["time_sec"] + offset
    index = np.searchsorted(ft, times, side="right") - 1
    valid = index >= 0
    index = np.clip(index, 0, len(frames) - 1)
    position = frames["position_beat"][index].astype(float)
    period = frames["beat_period"][index].astype(float)
    waiting = frames["waiting"][index]
    elapsed = times - ft[index]
    advance = ~waiting & (elapsed > 0) & (period > 0)
    position = np.where(advance, position + elapsed / np.where(period > 0, period, 1.0), position)
    return times[valid], position[valid], period[valid]
