# -*- coding: utf-8 -*-
"""Publish the avatar's live input streams over UDP.

The avatar runs in a separate process (pyanoduo, Python 3.13, with the policy
and Momentum), because it cannot share this interpreter. The one thing it
needs is what `accompanion/avatar.py` defines: the two event streams the
ACCompanion produces about its own playing, plus the schedule of its part.
This module serializes those as small JSON datagrams and sends them to the
avatar process, which reconstructs the same clock and schedule on the other
side (`pyanoduo.modules.aura_realtime`).

Only the standard library is used, so importing this never pulls in anything
the live ACCompanion does not already need. Publishing is off unless a host is
given, so a normal run is unaffected.

Wire it in three places (all guarded by ``if self.avatar_publisher``):

* the sequencer, at each ``note_on`` / ``note_off`` it sends -> :meth:`note`;
* ``follow_step``, once per frame (including the fermata wait) -> :meth:`frame`;
* ``accompaniment_step``, when the upcoming notes are (re)scheduled ->
  :meth:`schedule`.

The field names match `avatar.py`'s ``NOTE_EVENT_DTYPE`` / ``FRAME_EVENT_DTYPE``
and the avatar side's ``SCHEDULE_DTYPE``.
"""
import json
import socket


class AvatarPublisher(object):
    """Send note, frame, and schedule events to the avatar process over UDP.

    Parameters
    ----------
    host : str or None
        The avatar process's address. ``None`` disables publishing entirely
        (every method becomes a no-op), so the ACCompanion runs unchanged.
    port : int
        The avatar process's UDP port (its ``--in-port``, default 7001).
    """

    def __init__(self, host=None, port=7001):
        self.enabled = host is not None
        self.addr = (host, int(port))
        self._sock = (
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.enabled else None
        )

    def _send(self, payload):
        if not self.enabled:
            return
        try:
            self._sock.sendto(json.dumps(payload).encode("utf-8"), self.addr)
        except OSError:
            # A dropped datagram must never disturb the performance.
            pass

    def start(self, time_sec=0.0):
        """Announce the start of a performance (optional, for the avatar's logs)."""
        self._send({"type": "start", "time_sec": float(time_sec)})

    def stop(self):
        """Announce the end of a performance."""
        self._send({"type": "stop"})

    def note(self, note, on, time_sec):
        """One MIDI message the sequencer sends for the ACCompanion's part.

        `note` is an `accompanion.accompanist.score.Note`; call with ``on=True``
        where the sequencer sends ``note_on`` and ``on=False`` for ``note_off``,
        using the sequencer's ``c_time``.
        """
        self._send(
            {
                "type": "note",
                "time_sec": float(time_sec),
                "on": bool(on),
                "pitch": int(note.pitch),
                "velocity": int(note.velocity),
                "note_id": str(getattr(note, "id", "")),
                "score_onset_beat": float(note.onset),
                "score_duration_beat": float(note.duration),
            }
        )

    def frame(self, time_sec, position_beat, beat_period, waiting):
        """The following frame at the end of one ``follow_step``.

        ``position_beat`` is ``FollowingState.expected_position``, ``beat_period``
        the tempo model's, ``waiting`` the fermata hold's, ``time_sec`` the
        frame's ``solo_p_onset``.
        """
        self._send(
            {
                "type": "frame",
                "time_sec": float(time_sec),
                "position_beat": float(position_beat),
                "beat_period": float(beat_period),
                "waiting": bool(waiting),
            }
        )

    def schedule(self, notes, offset=0.0):
        """The ACCompanion's current plan of its part (``AccompanimentScore.notes``).

        Every note with a performed onset assigned so far, on the performance
        clock plus ``offset``. Sending it lets the avatar's look-ahead match
        training; without it the avatar still works from played notes alone.
        """
        rows = []
        for note in notes:
            onset = getattr(note, "p_onset", None)
            if onset is None:
                continue
            duration = getattr(note, "p_duration", None) or 0.0
            rows.append(
                [
                    str(note.id),
                    int(note.pitch),
                    float(onset) + float(offset),
                    float(duration),
                    int(note.velocity),
                ]
            )
        self._send({"type": "schedule", "time_sec": float(offset), "notes": rows})
