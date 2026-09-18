# The avatar input files

What the `avatar_<player>.npz` and `avatar-offline-self_<player>.npz` files in
`aura-data` contain, and how to get the individual inputs out of them.

Every file describes **one performance of the secondo part** — the part the
avatar will play — as the ACCompanion would see it while playing that part
itself: two event streams (the notes it sends, and its score position and
tempo per frame), the score of the part, and a feature vector per motion
frame computed from the streams. Live, the ACCompanion produces the same two
streams; the feature vector is the same function of them. Everything is
defined in `accompanion/avatar.py` in the ACCompanion repository.

There is one file per take, for the player whose `part` is `secondo` in
`metadata-aura.csv` — `avatar_p2.npz` in the `L1` takes, `avatar_p1.npz` in
the `L2` takes. The `avatar` column of the metadata gives the path.

## Contents of one file

All times are seconds in the MIDI file's clock, which is taken to be the
motion capture's clock as well (`frame_index / fps` in pyanoduo). Shapes
below are from `D05/B1_T1_L1/avatar_p2.npz`.

| array | shape | what it is |
|---|---|---|
| `note_events` | (554,) structured | The note stream: one `note_on` and one `note_off` per performed note (277 notes). |
| `frame_events` | (5156,) structured | The frame stream: score position and tempo every 10 ms, from 0.5 s before the primo's first note to the end. |
| `score_notes` | (291,) structured | The secondo part: every note of the score with its id, pitch, onset and duration in beats. |
| `skipped_note_ids` | (16,) str | Ids of score notes this player never played (the alignment's *deletions*). Kept out of the lookahead, see below. |
| `timestamps` | (1979,) float | The motion clock the features are computed on: 0, 1/30, 2/30, … s, up to 1 s past the last event. |
| `features` | (1979, 224) float32 | One feature vector per timestamp. |
| `feature_names` | (224,) str | The name of every feature column, in order. |
| `position_source` | scalar str | `online-primo` or `offline-self` — where `frame_events` came from. |
| `position_file` | scalar str | The tempo-curve file the frames were taken from. |
| `lookahead`, `recency_tau`, `horizon_sec`, `beats_per_bar` | scalars | The featurizer's parameters (8, 0.25 s, 5 s, 4). |

## The note stream: `note_events`

One row per MIDI message of the part. Live, the ACCompanion's sequencer
sends exactly these messages, and knows the score note behind each of them.

| field | type | meaning |
|---|---|---|
| `time_sec` | float | When the message was sent. |
| `on` | bool | `True` for `note_on`, `False` for `note_off`. |
| `pitch` | int | MIDI pitch, 21–108. |
| `velocity` | int | MIDI velocity for `note_on`; 0 for `note_off`. |
| `score_onset_beat` | float | Onset of the score note this message plays, in beats. `NaN` if the note is not in the score. |
| `score_duration_beat` | float | Its notated duration in beats. `NaN` if not in the score. |
| `note_id` | str | The score note's id (`p0n152`), matching `score_notes["note_id"]`. Empty if not in the score. |

Offline, the messages come from the recorded MIDI file and the score note from
the match file. A performed note the alignment could not place (an
*insertion*; 2 here, median 6 per take) keeps its `note_on`/`note_off` with an
empty score note: the player's hands moved for it, so the motion shows it.
Live this never happens — the ACCompanion only plays score notes.

## The frame stream: `frame_events`

One row per following frame, every 10 ms (the ACCompanion's polling period).

| field | type | meaning |
|---|---|---|
| `time_sec` | float | Time of the frame. |
| `position_beat` | float | Score position in beats: where in the piece the music is right now. |
| `beat_period` | float | Current tempo as seconds per beat (60 / bpm). |
| `waiting` | bool | `True` while the ACCompanion is holding at a fermata, waiting for the soloist. The position stands still meanwhile. |

Where the frames come from is the one difference between the two files:

| | `avatar_*.npz` (`online-primo`) | `avatar-offline-self_*.npz` (`offline-self`) |
|---|---|---|
| position of | the **primo** of the same take, as estimated **live** by the ACCompanion's score follower and tempo model while following it | this secondo performance, from its own **offline** alignment (ground truth) |
| character | causal; starts at the configured 60 bpm and converges over the first ~10 onsets; lags after a tempo change; occasionally lost after dropped notes (see `tempo_curves.json` in the take folder) | exact at every matched onset; tempo is the local median over ±2 beats |
| `waiting` | every frame the ACCompanion spent holding at the six fermatas | only while the player held longer than the tempo explains |
| this is what the avatar gets… | **live** — the ACCompanion follows the human primo, and its own notes follow that estimate | never; but it is the cleanest normalisation of the training data |

Between frames the position advances at the tempo, `position + elapsed /
beat_period`, and stands still while `waiting` — that is how the features
interpolate it.

## The feature vector: `features`, 224 columns

Every column is causal (uses only events at or before its timestamp) and in
physical units. Columns, by index:

| columns | name(s) | meaning |
|---|---|---|
| 0–87 | `key_down_21` … `key_down_108` | Velocity / 127 of every key currently held down by the part; 0 if not. Set by `note_on`, cleared by `note_off`. |
| 88–175 | `key_recency_21` … `key_recency_108` | `exp(−(t − last onset on that key) / 0.25 s)`: 1 at the moment a key is struck, 0.37 a quarter second later, ~0 after a second. Where the hands were a moment ago, with sub-frame timing. |
| 176 | `since_onset_sec` | Seconds since the part's last `note_on`, clipped at 5. |
| 177 | `to_next_onset_sec` | Seconds until the next score note, predicted as (its beat − position) × beat period. Clipped at 5. |
| 178 | `position_beat` | Score position, in beats (0 … 108 for the badinerie). |
| 179 | `beat_phase` | Position within the beat, 0 … 1. |
| 180 | `bar_phase` | Position within the bar (4 beats), 0 … 1. |
| 181 | `progress` | Share of the piece behind us, 0 … 1. |
| 182 | `beat_period_sec` | Current tempo, seconds per beat. |
| 183 | `waiting` | 1 while holding at a fermata, else 0. |
| 184–223 | `lookahead_0_valid` … `lookahead_7_duration_beat` | The next 8 score notes not yet played, nearest first, 5 numbers each: `valid` (1, or 0 when fewer than 8 notes remain), `pitch`, `delta_beat` (beats from the position to its onset), `delta_sec` (`delta_beat × beat_period`), `duration_beat`. Two notes of a chord take two slots. |

The lookahead is where the avatar's anticipation comes from: a pianist's hands
move towards the next notes before they sound. It never contains a note that
will not be played — live because the ACCompanion plays every note of its
part; offline because the notes this player skipped (`skipped_note_ids`) are
left out, and a note played more than a beat late is dropped from it and then
appears as a plain key press when it comes. So in both cases the same two
things hold: *every key press is a key press*, and *every note the lookahead
announces gets played*.

## Getting the inputs out

```python
import numpy as np

d = np.load("D05/B1_T1_L1/avatar_p2.npz")
d.files                                     # the arrays listed above
notes, frames = d["note_events"], d["frame_events"]
F, t, names = d["features"], d["timestamps"], list(d["feature_names"])
```

**The performed notes as a table** (one row per note, with its score position):

```python
table = []
for e in notes[notes["on"]]:
    # the note_off is the next off-message of the same pitch
    off = notes[~notes["on"] & (notes["pitch"] == e["pitch"]) & (notes["time_sec"] >= e["time_sec"])]
    table.append((e["time_sec"], off["time_sec"][0], int(e["pitch"]), int(e["velocity"]),
                  e["score_onset_beat"], str(e["note_id"])))
np.savetxt("notes.csv", np.array(table, dtype=object), fmt="%s", delimiter=",",
           header="onset_sec,offset_sec,pitch,velocity,score_onset_beat,note_id", comments="")
```

**The position / tempo curve**, and the beat position of any time (a motion
frame, say):

```python
np.savetxt("frames.csv", np.c_[frames["time_sec"], frames["position_beat"],
                               frames["beat_period"], frames["waiting"]],
           delimiter=",", header="time_sec,position_beat,beat_period,waiting")

beat_at = lambda seconds: np.interp(seconds, frames["time_sec"], frames["position_beat"])
beat_at(30.0)                                # e.g. 29.04
```

**One feature group as its own array**, by name:

```python
col = {n: i for i, n in enumerate(names)}
key_down    = F[:, col["key_down_21"]:col["key_down_108"] + 1]          # (frames, 88)
key_recency = F[:, col["key_recency_21"]:col["key_recency_108"] + 1]    # (frames, 88)
position    = F[:, col["position_beat"]]
tempo       = F[:, col["beat_period_sec"]]
lookahead   = F[:, col["lookahead_0_valid"]:].reshape(len(F), 8, 5)     # (frames, 8, 5)
```

**Everything as separate files** — one `.npy` per array, plus a CSV of the
features with named columns:

```python
import os
os.makedirs("D05_B1_T1_L1_p2", exist_ok=True)
for key in d.files:
    np.save(f"D05_B1_T1_L1_p2/{key}.npy", d[key])
np.savetxt("D05_B1_T1_L1_p2/features.csv", np.c_[t, F], delimiter=",",
           header="time_sec," + ",".join(names), comments="")
```

**The features on your own clock** — the motion frames' timestamps rather
than the stored 30 fps grid — recomputed from the streams (this is what a
pyanoduo `MidiConditioner` would do; the module needs only numpy):

```python
import sys; sys.path.insert(0, "/path/to/accompanion")     # the repository
from accompanion.avatar import AvatarFeaturizer

featurizer = AvatarFeaturizer(
    d["score_notes"],
    lookahead=int(d["lookahead"]), recency_tau=float(d["recency_tau"]),
    horizon_sec=float(d["horizon_sec"]), beats_per_bar=float(d["beats_per_bar"]),
    skipped_ids=d["skipped_note_ids"],
)
my_timestamps = motion.frame_indices / motion.fps          # pyanoduo's clock
X = featurizer.run(d["note_events"], d["frame_events"], my_timestamps)   # (len, 224)
```

**Warping the motion into score time**, so that a clip can be normalised by
tempo: give every motion frame its beat position from the frame stream, then
resample onto a regular beat grid.

```python
beats = beat_at(my_timestamps)                       # beat position of each motion frame
grid = np.arange(beats[0], beats[-1], 1 / 12)        # e.g. 12 samples per beat
pose_in_beat_time = np.stack([np.interp(grid, beats, pose[:, j]) for j in range(pose.shape[1])], 1)
```

With `avatar` this uses the ACCompanion's live estimate of the primo's
position; with `avatar-offline-self` the ground-truth position of the secondo
performance itself. During a fermata the estimate stands still (`waiting`),
so the held frames all map to the same beat — keep `waiting` as an input, or
cut the holds, rather than letting them collapse.

## Feeding pyanoduo

pyanoduo has two training paths, and these files serve both differently.

**The `dataset-midi-train` pipeline** (`pyanoduo train configs/midi.toml`)
encodes the music itself: it follows the primo MIDI with its own Matchmaker
clock (`PitchIOIHMM` + `KalmanTempoModel` + a continuous `ScoreClock`),
rasterises the secondo's notes on a beat grid (16 bins per beat, two beats
back and two ahead), and reads its data straight from the take folders --
`midi_p1.mid`, `midi_p2.mid`, `sync.csv` and the prepared secondo motion in
`video_<player>_top/`, with the MIDI score in `score/`. `aura-data` in the
nested layout is that contract, so point `root`, `motion_root` and `score`
at it. It maps roles as this folder does (L1: primo is p1; L2: primo is p2).
Its `sync.csv` places every file on a shared timeline; the ones here say
"the MIDI files' own clock, video at 0" (see `bin/write_aura_sync.py`).

Two boundaries in that pipeline are meant for the ACCompanion, and
`accompanion/avatar.py` provides the adapters:

- `AccompanimentSource.notes_at(time)` -- the secondo's currently scheduled
  notes. Offline that is the recording (with its future). Live it is the
  ACCompanion's own schedule: `AccompanionSchedule(acc_score.notes)`.
- `ScoreClock.observe(time, beat, beat_period)` -- the clock. Offline the
  pipeline runs its own follower over the primo; to train on what the
  ACCompanion will provide live instead, `clock_samples(frame_events,
  times)` turns the `frame_events` here into the `(times, beats, periods)`
  of pyanoduo's `ClockTrace`.

Running two followers on the same primo -- pyanoduo's and the
ACCompanion's -- would give the avatar and the accompaniment two slightly
different clocks; the adapters exist so that there is one.

**The original `MidiConditioner` path** (`run_experiment(config,
midi=...)`, with `[data] midi_enabled = true`) takes any per-frame feature
vector next to the partner's motion. `AvatarConditioner` in
`accompanion/avatar.py` serves the 224 features above to it:

```python
import sys; sys.path.insert(0, "/path/to/accompanion")
from accompanion.avatar import AvatarConditioner
from pyanoduo.modules.training import RunConfig, run_experiment

config = RunConfig.from_toml("configs/smoke.toml")          # with [data] root = aura-data,
run_experiment(config, midi=AvatarConditioner("aura-data")) # midi_enabled = true
```

and live, per frame, `RealtimeController.step(partner_pose, context,
midi=features_now)`. `bin/test_avatar_pipeline.py` runs this end to end
on the recordings that have motion; `bin/test_pyanoduo_midi.py` does the
same for the `dataset-midi-train` pipeline (needs its environment).

Two things to know on the motion side:

- That older loader (`load_recordings`) always takes `p2` as the target;
  the `dataset-midi-train` pipeline chooses by part. In the `L2` takes `p2`
  plays the primo, so only the newer pipeline handles them.
- The motion clock is `frame_index / fps` from the first video frame and is
  taken to be the MIDI clock. On `D05/B1_T1_L1` the two agree to within about
  half a second (the hands stop at the MIDI fermatas and start and finish
  with the notes); the videos' own epoch timestamps are not capture times
  and cannot be used to check this. Pin it down once against the raw video
  -- the frame of the first key press against the first MIDI onset -- and
  put the result in that take's `sync.csv`.

## Regenerating

From the ACCompanion repository, with the `aura-data` folder in either
layout:

    python bin/build_avatar_features.py --data ~/datasets/aura-data
    python bin/build_avatar_features.py --data ~/datasets/aura-data \
        --position offline-self --name avatar-offline-self

`--lookahead`, `--recency-tau`, `--horizon-sec`, `--fps` change the
parameters; `--keep-deleted` keeps skipped notes in the lookahead. The
streams themselves come from the match files and the tempo curves
(`bin/extract_tempo_curves.py`), so after editing an alignment in parangonada
and writing it back (`check_matchfiles.py`), re-run the tempo curves and then
this.
