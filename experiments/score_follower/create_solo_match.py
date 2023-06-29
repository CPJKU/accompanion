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
    if not os.path.exists(os.path.join(par_path, "match_solo", piece_name)):
        os.makedirs(os.path.join(par_path, "match_solo", piece_name))

    part = pt.load_score(score_path)[0]
    ppart, alignment = pt.load_match(match_path)
    perf_array = pt.musicanalysis.encode_performance(part, ppart[0], alignment)[0]
    note_array = part.note_array()
    a_matched_idx = np.array([d["score_id"] for d in alignment if d["label"] == "match"])
    matched_idx = np.nonzero(np.isin(note_array["id"], a_matched_idx))
    for i in range(5):
        new_perf_array = np.zeros((len(note_array)), np.dtype(
            [('beat_period', 'f4'), ('timing', 'f4'), ('articulation_log', 'f4'), ('velocity', 'f4')]))
        new_perf_array[matched_idx] = perf_array
        nm_idx = np.where(new_perf_array["beat_period"] == 0)[0]
        new_perf_array[nm_idx]["beat_period"] = perf_array["beat_period"].mean()
        new_perf_array[nm_idx]["velocity"] = perf_array["velocity"].mean()
        new_perf_array[nm_idx]["articulation_log"] = perf_array["articulation_log"].mean()
        # Add noise to the solo match.
        # new_perf_array["beat_period"] = new_perf_array["beat_period"] + np.random.normal(0, 0.01, len(part.note_array()))
        # new_perf_array["velocity"] = new_perf_array["velocity"] + np.random.normal(0, 0.01, len(part.note_array()))
        new_perf_array["velocity"][np.isnan(new_perf_array["velocity"])] = 0.5
        # new_perf_array["articulation_log"] = new_perf_array["articulation_log"] + np.random.normal(0, 0.01, len(part.note_array()))
        new_ppart, new_alignment = pt.musicanalysis.decode_performance(part, new_perf_array, return_alignment=True)
        for note in new_ppart.notes:
            if note["note_off"] - note["note_on"] < 0:
                note["note_off"] = note["note_on"]
        save_path = os.path.join(par_path, "match_solo", piece_name, piece_name + "_0" + str(i) + ".match")
        pt.save_match(new_alignment, new_ppart, part, save_path, assume_unfolded=True)
        del new_perf_array, new_ppart, new_alignment
        gc.collect()