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
    # Used in HMMACCompanion
    "gumbel_transition_matrix_scale": 0.5,
    "DECAY_VALUE": 1.0,
    "MIDI_KEYS": 88,
}
