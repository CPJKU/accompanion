import os, platform, argparse
import numpy as np
import pandas as pd
import partitura as pt


def get_beat_delay(gt_ppart, score, gt_alignment, time_delays):
    pnote_array = gt_ppart.note_array()
    note_array = score.note_array()
    pred_beats = time_delays.index.to_numpy()
    s_beat_delays = list()
    for idx, p_onset in enumerate(time_delays["Solo Performance Onset"].to_numpy()):
        p_id = pnote_array[np.argmin(np.abs(pnote_array["onset_sec"] - p_onset))]["id"]
        for a in gt_alignment:
            if a["label"] == "match":
                if a["performance_id"] == p_id:
                    onset = note_array[note_array["id"] == a["score_id"]]["onset_beat"]
                    s_beat_delays.append(abs(pred_beats[idx] - onset))
                    break
    s_beat_delays = np.array(s_beat_delays)
    return s_beat_delays.mean()


def get_time_delay(gt_ppart, score, gt_alignment, time_delays):
    pnote_array = gt_ppart.note_array()
    note_array = score.note_array()
    pred_onset = time_delays["Solo Performance Onset"].to_numpy()
    p_time_delays = list()
    for idx, s_onset in enumerate(time_delays.index.to_numpy()):
        s_idx = note_array[note_array["onset_beat"] == s_onset]["id"]
        tmp = []
        for s_id in s_idx:
            for a in gt_alignment:
                if a["label"] == "match":
                    if a["score_id"] == s_id:
                        onset = pnote_array[pnote_array["id"] == a["performance_id"]]["onset_sec"]
                        tmp.append(abs(pred_onset[idx] - onset))
        if tmp:
            p_time_delays.append(min(tmp))
    p_time_delays = np.array(p_time_delays)
    return p_time_delays.mean()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--follower", help="Which score follower to use", default="hmm")
    args = argparser.parse_args()

    pl = platform.system()
    if pl == "Windows":
        path_par = os.path.normpath("C:\\Users\\melki\\Desktop\\JKU\\data\\accompanion_experiment\\")
    else:
        path_par = os.path.normpath("/home/manos/Desktop/JKU/data/accompanion_experiment/")

    follower = args.follower
    artifact_path = os.path.join(os.path.dirname(__file__), "artifacts")
    beat_delays = []
    time_delays = []

    # Read the ground truth match
    for file in os.listdir(os.path.join(path_par, "match")):
        match_path = os.path.join(path_par, "match", file)
        gt_ppart, gt_alignment, score = pt.load_match(match_path, create_score=True)
        name = os.path.splitext(file)[0]
        time_delays = pd.read_csv(os.path.join(artifact_path, f"{name}_{follower}_time_delays.csv"), index_col=0, sep=",")
        mean_time = get_time_delay(gt_ppart, score, gt_alignment, time_delays)
        time_delays.append(mean_time)
        mean_beat = get_beat_delay(gt_ppart, score, gt_alignment, time_delays)
        beat_delays.append(mean_beat)

    print(f"Mean time delay: {np.mean(time_delays)}")
    print(f"Mean beat delay: {np.mean(beat_delays)}")

