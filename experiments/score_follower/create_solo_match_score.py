import partitura as pt
import os
import numpy as np
import gc

# Input your own path here, i.e. /home/User/data/accompanion_experiment/
par_path = ""

for file in os.listdir(os.path.join(par_path, "musicxml")):
    piece_name = os.path.splitext(file)[0]
    # Load the score
    score_path = os.path.join(par_path, "musicxml", file)
    # Load the Match
    match_path = os.path.join(par_path, "match", piece_name + ".match")
    # save Path
    if not os.path.exists(os.path.join(par_path, "match_gen", piece_name)):
        os.makedirs(os.path.join(par_path, "match_gen", piece_name))

    part = pt.load_score(score_path)[0]
    ppart, alignment = pt.load_match(match_path)
    perf_array = pt.musicanalysis.encode_performance(part, ppart[0], alignment)[0]
    mean_beat_period = np.mean(perf_array["beat_period"])
    mean_articulation = perf_array[~np.isnan(perf_array["articulation_log"])]["articulation_log"].mean()
    mean_dynamics = np.mean(perf_array["velocity"])
    for i in range(5):
        new_perf_array = np.zeros((len(part.note_array())), np.dtype(
            [('beat_period', 'f4'), ('timing', 'f4'), ('articulation_log', 'f4'), ('velocity', 'f4')]))
        new_perf_array["beat_period"] = mean_beat_period + np.random.normal(0, 0.01, len(part.note_array()))
        new_perf_array["velocity"] = mean_beat_period + np.random.normal(0, 0.1, len(part.note_array()))
        new_perf_array["articulation_log"] = mean_beat_period + np.random.normal(0, 0.05, len(part.note_array()))
        new_ppart, new_alignment = pt.musicanalysis.decode_performance(part, new_perf_array, return_alignment=True)
        for note in new_ppart.notes:
            if note["note_off"] - note["note_on"] < 0:
                note["note_off"] = note["note_on"]
        save_path = os.path.join(par_path, "match_gen", piece_name, piece_name + "_0" + str(i) + ".match")
        pt.save_match(new_alignment, new_ppart, part, save_path, assume_unfolded=True)
        del new_perf_array, new_ppart, new_alignment
        gc.collect()