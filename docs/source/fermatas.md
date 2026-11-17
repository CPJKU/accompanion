# Fermatas and free sections

At a fermata the notated durations stop meaning anything: the soloist holds for
as long as they feel like, and how long that is differs from player to player
and from run to run. The ACCompanion waits with them. It plays the accompaniment
up to and including the fermata, holds that chord under the soloist for as long
as they hold theirs, and moves on the moment they play the next note — picking
the tempo up where it was, rather than reading the wait as an enormous beat
period.

A free section — a cadenza, an *ad libitum* bar, a passage marked *senza
misura* — is the same suspension of the beat spread over a span of the score,
and is handled the same way: every solo onset inside it becomes a point to wait
at, so the passage is taken one note at a time.

Both are read off the score by [partitura] and need no configuration. Fermatas
come from the notation directly. Free sections are guessed from the words in
the score (*cadenza*, *ad libitum*, *senza misura*, *a piacere*, *freely*, …),
running from the marking to the next word or tempo direction — a guess, because
scores do not mark where a free passage ends. What was found is printed at
startup:

```
Waiting for the soloist at 15 onsets: 84, 112, 237, 237.5, 238, 238.5, ...
  free section: beats 236 to 244
```

A fermata over a rest, or one on the final chord, is dropped: waiting is ended
by the soloist arriving at the next onset, and those have none.

This costs nothing on a piece with no fermatas, and on one with them it is the
difference between an accompaniment that waits and one that plays on alone.
Measured with `bin/test_accompaniment.py` on the Badinerie, whose six fermatas
the synthesised soloist holds for a second or three each (median error of the
accompaniment against where it should have sounded, over three seeds):

| score follower | counting through | waiting |
| --- | --- | --- |
| `PitchIOIHMM` | 6840 ms early | 5 ms |
| `outerhmm` | 1 ms, but a quarter of the piece more than 200 ms out | 0 ms, none out |
| `arzt` | 3164 ms early | 36 ms |

Note that the score followers themselves track these performances perfectly
(`bin/test_followers.py --scenarios fermata` reports no error at all): what
fails without this is not the following but the accompanist's dead reckoning
between onsets.

To turn it off, or to correct what the score says, put a `fermata_kwargs` block
in the config file:

```yaml
    fermata_kwargs:
        enabled: True             # False counts through fermatas as before
        detect_free_sections: True  # read free sections from the score's words
        free_sections: [[236, 244]] # extra free spans, in score beats
        fermata_onsets: [25, 27]    # extra points to wait at, in score beats
        max_silence: 5.0          # give up waiting after this much silence
        max_lost: 0.4             # ...or this long after the soloist plays on
                                  #    without the follower reporting it
```

Both timeouts are there so that a wait can never hang the piece: whichever
expires first, the accompaniment resumes in tempo, which is exactly what it
would have done without any of this. `max_silence` covers a soloist who stops
altogether; `max_lost` covers a soloist who plays on while the score follower
loses them. Waiting makes the accompaniment purely reactive at every point it
waits at, so on a piece with a long run of fermatas played strictly in time,
`enabled: False` is the safer setting.

To hear the difference offline:

```shell
python ./bin/test_accompaniment.py --piece badinerie --scenarios fermata
python ./bin/test_accompaniment.py --piece badinerie --scenarios fermata --no-fermata
```

[partitura]: https://github.com/CPJKU/partitura
