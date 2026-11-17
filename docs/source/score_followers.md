# Score followers

The ACCompanion follows the soloist with a score follower from
[matchmaker](https://github.com/pymatchmaker/matchmaker). Which one is used is
set by two config keys (or the equivalent command line flags): `follower`
chooses the ACCompanion variant, `score_follower` the tracker itself.

| `follower` | `score_follower` | |
| --- | --- | --- |
| `hmm` | `PitchIOIHMM`, `PitchIOIKHMM` | Hidden Markov models over pitch and inter-onset intervals. Best for a monophonic solo part. |
| `oltw` | `OnlineTimeWarping` | On-line time warping against one or more reference performances. Best for dense, polyphonic parts. |
| `matchmaker` | any matchmaker method | Every other score follower matchmaker provides. |

The first two are wired up with the ACCompanion's own tuning, and are what the
config files in `config_files/` use. The `matchmaker` variant instead builds
whatever matchmaker's registry declares, so score followers added to matchmaker
later are usable without any change here:

```shell
python ./bin/launch_acc.py --follower matchmaker --score-follower arzt \
    --input Your_MIDI_Input --output Your_MIDI_Output
```

To list every score follower available in your installation:

```shell
python ./bin/launch_acc.py --list-followers   # names --score-follower accepts
python ./bin/test_followers.py --list         # the same, plus the test scenarios
```

## Comparing score followers

`bin/test_followers.py` runs a performance through the followers the same way
the ACCompanion's main loop does, but offline: no MIDI hardware, no audio, no
accompaniment. Use it to see which followers work on a piece and how closely
they track it.

```shell
# every follower, on a performance played exactly as written
python ./bin/test_followers.py --piece bach_menuett

# every follower, on realistic playing: bent tempo, smeared timing, wrong notes
python ./bin/test_followers.py --piece bach_menuett --scenarios all --seeds 3

# a few followers, on a real MIDI recording
python ./bin/test_followers.py --piece bach_menuett --followers hmm arzt OPTM \
    --midi-fn path/to/performance.mid
```

The performance is synthesised from the solo score, so the true score position
is known at every instant — which is what makes the error and latency columns
meaningful. It can be synthesised badly on purpose; `--scenarios` picks how the
part is played:

| scenario | |
| --- | --- |
| `clean` | exactly as written, at a constant tempo |
| `rubato` | tempo bent ±15% over 8-beat phrases, slowing 35% into the final bars |
| `jitter` | onsets scattered by 30 ms, chords rolled over 25 ms |
| `errors` | 5% of notes at the wrong pitch, 5% dropped, 3% spurious extras |
| `human` | all of the above, milder — roughly an amateur run-through |
| `fermata` | otherwise metronomic, but every fermata and free section is taken in the player's own time |
| `freehand` | `human`, with those same holds |

`clean` is deterministic; the others are averaged over `--seeds` random draws.
Ranking followers on `clean` alone is misleading: it measures whether a
follower stays locked, not whether it recovers, and the ranking does change
under the other scenarios.
