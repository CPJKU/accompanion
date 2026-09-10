# -*- coding: utf-8 -*-
CONFIG = {
    # Found in base.py
    "ACC_PROCESS": True,
    # Found in base.py
    "USE_THREADS": True,
    # Ritenuto Length for Accompnist in Accopanist decoder
    "RIT_LEN": 24,
    # Ritenuto Window Length found in  Accompanist in Accopanist decoder
    "RIT_W": 0.75,
    # Ritenuto curvature found in Accompanist decoder
    "RIT_Q": 2.0,
    # I/O MIDI
    "BACKEND": "mido",
    "POLLING_PERIOD": 0.02,
    # Used in HMMACCompanion. Dispersion of the HMM's state transition
    # distribution: how far ahead the follower is willing to jump in one step.
    # 0.5 is too narrow to skip over a note the player left out -- the follower
    # then falls behind and never recovers. Measured with
    # bin/test_accompaniment.py --piece badinerie --scenarios all: going from
    # 0.5 to 1.0 leaves a clean performance untouched, and under realistic
    # playing takes the accompaniment from 2.1s late to 59ms, and from 16% of
    # it never sounding at all to none. 2.0 is already too wide -- it misplaces
    # 78% of the accompaniment on a clean run-through, which the follower's own
    # error does not show.
    "gumbel_transition_matrix_scale": 1.0,
    "DECAY_VALUE": 1.0,
    "MIDI_KEYS": 88,
}
