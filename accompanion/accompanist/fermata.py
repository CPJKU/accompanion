# -*- coding: utf-8 -*-
"""
Waiting with the soloist.

At a fermata the notated durations stop meaning anything: the soloist holds
for as long as they feel like holding, and an accompanist holds with them
rather than counting on. A free section -- a cadenza, an *ad libitum* bar, a
passage marked *senza misura* -- is the same suspension of the beat, spread
over a span of the score instead of concentrated on one chord.

Both are handled here by one mechanism. They are resolved into *waiting
points*: score onsets at which the accompaniment stops and waits to be moved
on. A fermata contributes the onset it sits on; a free section contributes
every solo onset it covers, so the passage is taken one note at a time.

At a waiting point `FermataHold` parks every accompaniment note the
accompanist has scheduled beyond it, keeps the chord sounding under the
soloist for as long as the wait lasts, and lets go when the score follower
reports the next onset -- or, if the soloist has fallen silent for longer
than a fermata can plausibly last, gives up and resumes in tempo rather than
leaving the piece hanging.
"""
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Score onsets closer than this are the same onset. Score positions reach
#: here from several directions -- a `beat_map` in double precision, a note
#: array in single -- and the smallest interval anyone notates is orders of
#: magnitude larger than this.
ONSET_TOLERANCE = 1e-3

#: Notes closer together than this are one attack: a rolled chord, or a pair of
#: hands that do not quite agree. A note that follows a gap this long is a new
#: attack, and during a wait that means the soloist has left the fermata --
#: whereas the tail of the fermata's own chord, arriving a few frames after
#: the wait began, does not.
WAIT_SETTLE = 0.1

#: Words that mark the start of a passage played out of time. Matched as
#: substrings, case insensitively, against the text of every word and tempo
#: direction in the score.
FREE_SECTION_WORDS = (
    "cadenza",
    "ad lib",
    "ad libitum",
    "senza misura",
    "senza tempo",
    "a piacere",
    "colla parte",
    "col canto",
    "freely",
    "free",
    "improvis",
)


def _marking_text(marking) -> str:
    """The text of a partitura `Words` or `Direction`, or ``""``."""
    for attribute in ("text", "raw_text"):
        value = getattr(marking, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_free_marking(text: str, words: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def fermata_onsets_from_part(part) -> np.ndarray:
    """The score onsets, in beats, that carry a fermata.

    Parameters
    ----------
    part : partitura.score.Part
        Anything else -- a `PerformedPart`, say, which has no fermatas to
        report -- yields an empty result rather than an error.

    Returns
    -------
    np.ndarray
        Sorted, unique score positions in beats. A fermata over a rest or a
        barline is included: whether anything is played there is decided
        later, against the onsets the solo part actually has.
    """
    beat_map = getattr(part, "beat_map", None)
    iter_all = getattr(part, "iter_all", None)
    if beat_map is None or iter_all is None:
        return np.array([], dtype=float)

    from partitura.score import Fermata

    onsets = [float(beat_map(fermata.start.t)) for fermata in iter_all(Fermata)]
    return np.unique(onsets) if onsets else np.array([], dtype=float)


def free_sections_from_part(
    part,
    words: Sequence[str] = FREE_SECTION_WORDS,
) -> List[Tuple[float, float]]:
    """The spans of the score, in beats, marked to be played out of time.

    A marking such as *cadenza*, *ad libitum* or *senza misura* says where a
    free passage starts but not where it ends; scores leave that to the next
    marking that re-establishes a tempo or a character. So a free section is
    read as running from its own marking to the next word or tempo direction
    that is not itself a free marking, or to the end of the part.

    Loudness and articulation directions are ignored: a *piano* in the middle
    of a cadenza does not end it.

    Returns
    -------
    list of (float, float)
        ``(start, end)`` pairs in beats, sorted and non-overlapping. The end
        is exclusive.
    """
    beat_map = getattr(part, "beat_map", None)
    iter_all = getattr(part, "iter_all", None)
    if beat_map is None or iter_all is None:
        return []

    from partitura.score import TempoDirection, Words

    markings = []
    for cls in (Words, TempoDirection):
        for marking in iter_all(cls, include_subclasses=True):
            text = _marking_text(marking)
            if text:
                markings.append((float(beat_map(marking.start.t)), text))
    markings.sort(key=lambda m: m[0])

    end_of_part = _part_end_beat(part, beat_map)

    sections = []
    for i, (start, text) in enumerate(markings):
        if not _is_free_marking(text, words):
            continue
        end = end_of_part
        for position, other in markings[i + 1 :]:
            if position > start + ONSET_TOLERANCE and not _is_free_marking(
                other, words
            ):
                end = position
                break
        sections.append((start, end))

    return merge_sections(sections)


def _part_end_beat(part, beat_map) -> float:
    last_point = getattr(part, "last_point", None)
    if last_point is not None:
        return float(beat_map(last_point.t))
    note_array = part.note_array()
    return float((note_array["onset_beat"] + note_array["duration_beat"]).max())


def merge_sections(
    sections: Iterable[Sequence[float]],
) -> List[Tuple[float, float]]:
    """Sort `(start, end)` spans and fuse the ones that touch or overlap."""
    ordered = sorted(
        (float(start), float(end)) for start, end in sections if float(end) > float(start)
    )
    merged: List[Tuple[float, float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1] + ONSET_TOLERANCE:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def waiting_points(
    solo_onsets: np.ndarray,
    fermata_onsets: Iterable[float] = (),
    free_sections: Iterable[Sequence[float]] = (),
) -> np.ndarray:
    """The solo score onsets at which the accompaniment waits.

    A fermata is kept only if the solo part has an onset there, since a
    waiting point is recognised by the soloist arriving at it: a fermata over
    a rest, or one carried by a voice that is not the solo part, has nothing
    to wait for and is dropped. A free section contributes every solo onset it
    covers except its last, which is where the passage is left rather than
    waited at.

    Parameters
    ----------
    solo_onsets : np.ndarray
        The unique score onsets of the solo part, in beats.
    fermata_onsets : iterable of float
    free_sections : iterable of (float, float)
        Half-open spans in beats.

    Returns
    -------
    np.ndarray
        Sorted, unique solo onsets, as members of `solo_onsets`.
    """
    onsets = np.asarray(solo_onsets, dtype=float)
    if len(onsets) == 0:
        return np.array([], dtype=float)

    points = []
    for fermata in fermata_onsets:
        matches = onsets[np.isclose(onsets, float(fermata), atol=ONSET_TOLERANCE)]
        if len(matches) > 0:
            points.append(float(matches[0]))

    for start, end in free_sections:
        inside = onsets[(onsets >= float(start) - ONSET_TOLERANCE) & (onsets < float(end))]
        points.extend(float(onset) for onset in inside)

    if not points:
        return np.array([], dtype=float)

    # The last onset of the piece is no place to wait: nothing follows that
    # could release the hold, so it would only be held until the timeout.
    return np.unique([point for point in points if point < onsets[-1]])


class FermataHold(object):
    """Holds the accompaniment at a waiting point until the soloist moves on.

    The hold works entirely through the onset times the accompanist assigns to
    the accompaniment notes, which is the same handle the accompanist itself
    uses and the only thing the sequencer reads. Held notes are parked at an
    infinite onset, so the sequencer passes over them without ever deciding
    the piece has ended, and are given real onsets again when the hold is
    released.

    Parameters
    ----------
    acc_notes : iterable of accompanion.accompanist.score.Note
        The notes of the accompaniment score, whose onset times this object
        parks and restores.
    onsets : iterable of float
        The waiting points, in score beats, as returned by `waiting_points`.
        An empty set makes every method a no-op.
    solo_onsets : iterable of float
        Every unique onset of the solo part, in beats. Only used to know how
        far the onset that will end a wait is from the waiting point, which is
        what `score_gap` reports.
    max_silence : float
        How long the soloist may be silent -- no key down, no note played --
        before a wait is taken for a breakdown rather than a held fermata.
        The accompaniment then resumes in tempo, which is what it would have
        done all along without this class.
    max_lost : float
        The other way a wait ends badly: the soloist plays on, but the score
        follower never reports the onset they moved to, so nothing releases
        the hold. This is how long the accompaniment goes on waiting after
        hearing them attack a note that is not the fermata's own, before
        giving the wait up as a lost position rather than a held note. It
        needs only to cover the follower's own latency; without it, the
        accompaniment can wait out the rest of the piece in silence.
    sustain_margin : float
        How far ahead of the clock the note-off of a held chord is kept, in
        seconds. Large enough that the chord cannot be cut between two input
        frames, small enough not to blur into what follows the fermata.
    verbose : bool
        Print each wait as it starts and ends, in the manner of the rest of
        the following loop.
    """

    def __init__(
        self,
        acc_notes: Iterable,
        onsets: Iterable[float] = (),
        solo_onsets: Iterable[float] = (),
        max_silence: float = 5.0,
        max_lost: float = 0.4,
        sustain_margin: float = 0.25,
        verbose: bool = True,
    ) -> None:
        self.notes = sorted(acc_notes, key=lambda note: note.onset)
        self.onsets = np.asarray(sorted(onsets), dtype=float)
        self.solo_onsets = np.asarray(sorted(solo_onsets), dtype=float)
        self.max_silence = float(max_silence)
        self.max_lost = float(max_lost)
        self.sustain_margin = float(sustain_margin)
        self.verbose = verbose

        #: The score onset currently being waited at, or None.
        self.onset: Optional[float] = None
        #: ``(score onset, seconds waited)`` for every wait so far.
        self.waits: List[Tuple[float, float]] = []

        self._held: List = []
        self._sustained: List[Tuple[object, float]] = []
        self._score_gap: float = 0.0
        self._began_at: Optional[float] = None
        self._last_active: Optional[float] = None
        self._playing_since: Optional[float] = None
        self._last_attack: Optional[float] = None
        self._gave_up: Optional[str] = None

    def __len__(self) -> int:
        return len(self.onsets)

    @property
    def waiting(self) -> bool:
        """Whether the accompaniment is currently stopped at a waiting point."""
        return self.onset is not None

    @property
    def score_gap(self) -> float:
        """Beats from the current waiting point to the onset that will end it.

        However long the wait turns out to last, this is the interval the
        score writes between the fermata and what follows it -- the one a
        score follower's timing model should be shown, in place of the wait
        the clock actually measured.
        """
        return self._score_gap

    def waits_at(self, score_onset: float) -> bool:
        """Whether `score_onset` is a waiting point."""
        if len(self.onsets) == 0:
            return False
        return bool(
            np.any(np.isclose(self.onsets, float(score_onset), atol=ONSET_TOLERANCE))
        )

    def begin(self, score_onset: float, perf_onset: float) -> bool:
        """Stop the accompaniment at `score_onset`, if it is a waiting point.

        Everything the accompanist has scheduled beyond this onset is parked;
        everything sounding at it is kept sounding by `keep_waiting`. Call
        this after the accompaniment step for the onset, so that the notes on
        the fermata itself are played with the tempo the soloist arrived at,
        and only what comes after is suspended.

        Parameters
        ----------
        score_onset : float
            The score onset the soloist has just reached, in beats.
        perf_onset : float
            When they reached it, in seconds since the performance started.

        Returns
        -------
        bool
            Whether a wait was started.
        """
        if self.waiting or not self.waits_at(score_onset):
            return False

        self.onset = float(score_onset)
        later = self.solo_onsets[self.solo_onsets > self.onset + ONSET_TOLERANCE]
        self._score_gap = float(later[0] - self.onset) if len(later) else 0.0
        self._began_at = perf_onset
        self._last_active = perf_onset
        self._playing_since = None
        # The fermata's own chord landed on this frame; anything that follows
        # it closely enough is the rest of that chord.
        self._last_attack = perf_onset
        self._held = []
        self._sustained = []

        for note in self.notes:
            if note.onset > self.onset + ONSET_TOLERANCE:
                if not note.already_performed:
                    note.p_onset = np.inf
                    self._held.append(note)
            elif (
                note.p_duration is not None
                and note.onset + note.duration > self.onset + ONSET_TOLERANCE
            ):
                # Sounding across the waiting point, so it holds with the
                # soloist instead of stopping after its notated value.
                self._sustained.append((note, float(note.p_duration)))

        if self.verbose:
            print(
                f"waiting at {self.onset} "
                f"({len(self._held)} notes held, {len(self._sustained)} sounding)"
            )
        return True

    def keep_waiting(
        self,
        perf_onset: float,
        holding: bool,
        new_notes: bool = False,
    ) -> bool:
        """Sustain the held chord for another frame, and say whether to go on.

        A wait is released from outside, by the soloist reaching the next
        onset. The two things this decides are the ways that never happens:
        the soloist stops playing altogether, or they play on and the score
        follower does not find them. Both end the wait, and in both cases the
        accompaniment resumes in tempo -- no worse off than an ACCompanion
        that never waited at all.

        Parameters
        ----------
        perf_onset : float
            The time of this input frame, in seconds.
        holding : bool
            Whether the soloist still has a key down. This, and not the
            elapsed time, is what says a fermata is still being held: a
            fermata lasts as long as it lasts.
        new_notes : bool
            Whether a note started in this frame. After a fermata's own chord
            has settled, a new note means the soloist has moved on, and the
            wait is living on borrowed time.

        Returns
        -------
        bool
            True while the accompaniment should keep waiting.
        """
        if not self.waiting:
            return False

        if holding or new_notes:
            self._last_active = perf_onset
        if new_notes:
            if (
                self._playing_since is None
                and perf_onset - self._last_attack > WAIT_SETTLE
            ):
                # A fresh attack, not the tail of the chord the wait began on.
                self._playing_since = perf_onset
            self._last_attack = perf_onset

        sustain_until = perf_onset + self.sustain_margin
        for note, _ in self._sustained:
            if note.p_onset is None or not np.isfinite(note.p_onset):
                continue
            if sustain_until > note.p_onset + note.p_duration:
                note.p_duration = sustain_until - note.p_onset

        if perf_onset - self._last_active > self.max_silence:
            self._gave_up = "silence"
            return False
        if (
            self._playing_since is not None
            and perf_onset - self._playing_since > self.max_lost
        ):
            self._gave_up = "lost"
            return False
        return True

    def resume(
        self,
        perf_onset: float,
        score_onset: float,
        beat_period: float,
    ) -> float:
        """Let go of the wait: the soloist has played `score_onset`.

        The accompaniment picks up as if the wait had taken no score time at
        all -- what was written on `score_onset` sounds now, and what follows
        it is spaced from here at the tempo of before the wait. The
        accompanist reschedules all of it on this same frame; these onsets
        only make sure nothing is left parked if it does not.

        Returns
        -------
        float
            How long the wait lasted, in seconds.
        """
        return self._release(perf_onset, float(score_onset), beat_period)

    def give_up(self, perf_onset: float, beat_period: float) -> float:
        """Abandon a wait the soloist has not ended, and resume in tempo.

        The accompaniment carries on from the waiting point at the tempo it
        had, which is what it would have done without any of this. Returns how
        long the wait lasted, in seconds.
        """
        if self.verbose and self.waiting:
            reason = (
                "the soloist has gone quiet"
                if self._gave_up == "silence"
                else "the soloist has moved on and the follower has not"
            )
            print(f"giving up the wait at {self.onset}: {reason}")
        return self._release(perf_onset, self.onset, beat_period)

    def _release(
        self,
        perf_onset: float,
        resume_onset: float,
        beat_period: float,
    ) -> float:
        if not self.waiting:
            return 0.0

        for note in self._held:
            # Score time resumes at `resume_onset`, so that is "now"; anything
            # the wait skipped past sounds immediately rather than being lost.
            ahead = max(0.0, float(note.onset) - resume_onset)
            note.p_onset = perf_onset + beat_period * ahead

        for note, notated in self._sustained:
            if note.p_onset is None or not np.isfinite(note.p_onset):
                continue
            # Let go when the soloist does, but never before the note has had
            # its notated length -- it may not even have sounded yet.
            note.p_duration = max(notated, perf_onset - note.p_onset)

        waited = perf_onset - self._began_at
        self.waits.append((self.onset, waited))
        if self.verbose:
            print(f"waited {waited:.2f}s at {self.onset}, resuming at {resume_onset}")

        self.onset = None
        self._score_gap = 0.0
        self._held = []
        self._sustained = []
        self._began_at = None
        self._last_active = None
        self._playing_since = None
        self._last_attack = None
        self._gave_up = None
        return waited
