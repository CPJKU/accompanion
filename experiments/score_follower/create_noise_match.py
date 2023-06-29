import copy
import partitura as pt
import os
import numpy as np
import gc


# Input your own path here, i.e. /home/User/data/accompanion_experiment/
par_path = ""

for file in os.listdir(os.path.join(par_path, "musicxml")):
    piece_name = os.path.splitext(file)[0]
    print(piece_name)
    # Load the score
    score_path = os.path.join(par_path, "musicxml", file)
    # Load the Match
    match_path = os.path.join(par_path, "match", piece_name + ".match")
    # save Path
    if not os.path.exists(os.path.join(par_path, "match_solo", piece_name)):
        os.makedirs(os.path.join(par_path, "match_solo", piece_name))

    mf = pt.io.importmatch.load_matchfile(match_path)
    mcr = mf.info("midiClockRate")
    mcu = mf.info("midiClockUnits")
    miliseconds = 100
    for i in range(5):
        umf = copy.deepcopy(mf)
        noise_onset = np.clip(pt.utils.music.seconds_to_midi_ticks(np.random.normal(0, miliseconds/1000, len(mf.notes)), mcr, mcu), -300, 300)
        noise_offset = np.clip(pt.utils.music.seconds_to_midi_ticks(np.random.normal(0, miliseconds/1000, len(mf.notes)), mcr, mcu), -300, 300)
        for idx, note in enumerate(umf.notes):
            note.Onset = note.Onset + noise_onset[idx]
            note.Offset = note.Offset + noise_offset[idx]
            if note.Offset - note.Onset < 0:
                note.Offset = note.Onset
        save_path = os.path.join(par_path, "match_solo", piece_name, piece_name + "_0" + str(i) + ".match")
        umf.write(save_path)
        del umf
        gc.collect()