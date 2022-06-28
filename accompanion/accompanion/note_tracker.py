import math
import numpy as np
from collections import defaultdict


class TrackedNote(object):
    def __init__(self, pitch=None, onset=None, duration=None, id=None):
        self.pitch = pitch
        self.onset = onset
        self.duration = duration
        self.id = id


class NoteTracker(object):
    __slots__ = [
        "score",
        "onset",
        "durations",
        "notes",
        "velocities",
        "open_notes",
        "matches",
        "already_matched",
        "time",
        "note_dict",
        "recently_closed_snotes"
    ]


    def __init__(self, score):
        self.score = score
        self.open_notes = dict()
        self.notes = []  # 2d list of pitches (multiple notes can have the same onset)
        self.velocities = []  # 2d list of velocities
        self.onset = []  # 1d list
        self.durations = []  # 2d list (multiple notes can have the same onset)
        self.matches = []  # 2d list of matched score note ids
        self.already_matched = []
        # 1d list of all matched score note ids (easier to check if id is already matched)
        self.time = 0
        self.recently_closed_snotes = []
        self.setup_tracked_notes()

    def setup_tracked_notes(self):
        self.note_dict = defaultdict(list)
        for note in self.score:
            # 0: pitch, 1: duration_beat, 2: onset, 3: durations,
            # 3: index in notes,
            # 4: index of the note in sublist
            self.note_dict[note['id']] += [
                note['pitch'],
                max(note['duration_beat'], 1 / 32),
                None, None, None, None
            ]


    def track_note(self, midi_msg):

        # time now is the absolute time of the message, not
        # delta time, since this time would not be correct in
        # case there are non-note messages (e.g., pedal, etc.)
        midi_type, note, velocity, time = midi_msg

        same_block = math.isclose(time, self.time)

        self.time = time

        if midi_type == "note_on" and velocity > 0:

            if not same_block or len(self.notes) == 0:
                self.durations.append([None])
                self.notes.append([note])
                self.velocities.append([velocity])
                self.onset.append(self.time)

            else:
                self.durations[-1].append(None)
                self.notes[-1].append(note)
                self.velocities[-1].append(velocity)

            self.open_notes[note] = (
                self.time,
                len(self.durations) - 1,
                len(self.durations[-1]) - 1,
            )

        elif midi_type == "note_off" or velocity == 0:
            if note in self.open_notes.keys():
                onset, durations_index, duration_id = self.open_notes[note]

                p_dur = self.time - onset
                self.durations[durations_index][duration_id] = p_dur

                try:
                    snote_id = self.matches[durations_index][duration_id]
                    if snote_id is not None:
                        self.note_dict[snote_id][3] = p_dur
                        self.note_dict[snote_id][4] = durations_index
                        self.note_dict[snote_id][5] = duration_id
                        self.recently_closed_snotes.append(snote_id)
                except IndexError:
                    pass
                del self.open_notes[note]
            else:
                pass
            # raise ValueError('Detected a note_off event, however no matching previous note_on event')

    def update_alignment(self, score_time):

        # apply time constraint
        score_window = (self.score["onset_beat"] <= score_time + 2) & (
            score_time - 2 <= self.score["onset_beat"]
        )
        candidates = self.score[score_window]

        # try to align the notes that came in last
        for idx in range(len(self.matches), len(self.notes)):
            notes = self.notes[idx]

            # find matching notes within score window
            matched_ids = []
            for note in notes:
                # apply pitch constraint for each note
                matching_pitch = candidates[candidates["pitch"] == note]

                score_id = None
                for match in matching_pitch:
                    if (
                        match["id"] not in matched_ids
                        and match["id"] not in self.already_matched
                    ):
                        score_id = match["id"]

                        self.already_matched.append(
                            score_id
                        )  # mark score note as matched
                        break

                if score_id is None:
                    print("No match for", note)
                else:
                    self.note_dict[score_id][2] = self.onset[idx]
                matched_ids.append(score_id)

            # store matches for particular notes
            self.matches.append(matched_ids)

    def export_midi(self, out_fn):

        # note_array = np.zeros(
        #     len(self.notes),
        #     dtype=[
        #         ('pitch', 'i4'),
        #         ('onset_sec', 'f4'),
        #         ('duration_sec', 'f4'),
        #         ('velocity', 'i4')
        #     ]
        # )

        ninfo = (
            self.notes,
            self.onset,
            self.durations,
            self.velocities
        )

        


            
