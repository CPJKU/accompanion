import os, platform, argparse
import numpy as np
import pandas as pd
import partitura as pt
import parangonar as pg


def get_beat_delay(gt_ppart, score, gt_alignment, time_delays):
    """
    Get the beat delay of the predicted performance

    Parameters
    ----------
    gt_ppart : partitura.performance.Performance
        The ground truth performance
    score : partitura.score.Part
        The score
    gt_alignment : list
        The ground truth alignment
    time_delays : pandas.DataFrame
        The time delays of the predicted performance
    """
    pnote_array = gt_ppart.note_array()
    note_array = score.note_array()
    pred_beats = time_delays.index.to_numpy()
    s_beat_delays = list()
    already_found = list()
    # Loop over the predicted onset times to select the closest score beat.
    for idx, p_onset in enumerate(time_delays["Solo Performance Onset"].to_numpy()):
        # Filter indices to not use same index twice
        rest_idx = ~np.isin(np.arange(len(pnote_array)), np.array(already_found))
        # Get the performance note indices that are closest to the predicted onset
        p_id = pnote_array[np.argmin(np.abs(pnote_array[rest_idx]["onset_sec"] - p_onset))]["id"]
        # Loop over the ground truth alignment to find the score note that matches the performance note
        for a in gt_alignment:
            # only loop over matches
            if a["label"] == "match":
                if a["performance_id"] == p_id:
                    # Get the score beat that corresponds to the score note
                    onset = note_array[note_array["id"] == a["score_id"]]["onset_beat"]
                    # Append the difference between the predicted beat and the score beat
                    s_beat_delays.append(abs(pred_beats[idx] - onset))
                    break
    # Convert to numpy array to calculate the mean
    s_beat_delays = np.array(s_beat_delays)
    if s_beat_delays.size == 0:
        return 0
    return s_beat_delays.mean()


def get_time_delay(gt_ppart, score, gt_alignment, time_delays):
    """
    Get the time delay of the predicted performance

    Parameters
    ----------
    gt_ppart : partitura.performance.Performance
        The ground truth performance
    score : partitura.score.Part
        The score
    gt_alignment : list
        The ground truth alignment
    time_delays : pandas.DataFrame
        The time delays of the predicted performance

    Returns
    -------
    float
        The mean time delay of the predicted performance
    """
    pnote_array = gt_ppart.note_array()
    note_array = score.note_array()
    pred_onset = time_delays["Solo Performance Onset"].to_numpy()
    p_time_delays = list()
    # Loop over the predicted beat times to select the closest performance onset sec.
    for idx, s_onset in enumerate(time_delays.index.to_numpy()):
        # Get the score note indices that match to the predicted beat
        s_idx = note_array[note_array["onset_beat"] == s_onset]["id"]
        tmp = []
        # If many indices per beat we need to select the minimum match
        for s_id in s_idx:
            # Loop over the ground truth alignment to find the performance note that matches the score note
            for a in gt_alignment:
                # only loop over matches
                if a["label"] == "match":
                    if a["score_id"] == s_id:
                        # Get the performance onset sec that corresponds to the performance note
                        onset = pnote_array[pnote_array["id"] == a["performance_id"]]["onset_sec"]
                        # Append the difference between the predicted onset sec and the performance onset sec
                        tmp.append(abs(pred_onset[idx] - onset))
        # Select the minimum difference if many indices per beat
        if tmp:
            p_time_delays.append(min(tmp))
    # Convert to numpy array to calculate the mean
    p_time_delays = np.array(p_time_delays)
    if p_time_delays.size == 0:
        return 0
    return p_time_delays.mean()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--follower", help="Which score follower to use", default="hmm")
    args = argparser.parse_args()

    # Input your own path here, i.e. /home/User/data/accompanion_experiment/
    path_par = ""

    follower = args.follower
    artifact_path = os.path.join(os.path.dirname(__file__), "artifacts")
    beat_delays = []
    second_delays = []
    fscores = []
    filenames = []
    # Read the ground truth match
    for file in os.listdir(os.path.join(path_par, "match")):
        match_path = os.path.join(path_par, "match", file)
        gt_ppart, gt_alignment, score = pt.load_match(match_path, create_score=True)
        name = os.path.splitext(file)[0]
        time_delays = pd.read_csv(os.path.join(artifact_path, f"{name}_{follower}_time_delays.csv"), index_col=0, sep=",")
        a = pd.read_csv(os.path.join(artifact_path, f"{name}_{follower}_alignment.csv"), sep=",")
        mean_time = get_time_delay(gt_ppart, score, gt_alignment, time_delays)
        second_delays.append(mean_time)
        mean_beat = get_beat_delay(gt_ppart, score, gt_alignment, time_delays)
        beat_delays.append(mean_beat)
        # Calculate the F-score we need to rename the columns to match the ground truth
        pred_alignment = [{"label": ("match" if row[1].matchtype == 0 else "insertion"), "score_id": row[1].partid,
                           "performance_id": row[1].ppartid[4:]} for row in a.iterrows()]

        # Load Ground Truth Alignment
        fmeasure = pg.fscore_alignments(pred_alignment, gt_alignment, types=["match"])[2]
        fscores.append(fmeasure)
        filenames.append(name)

    df = pd.DataFrame({"Filename": filenames, "Beat Delay": beat_delays, "Time Delay": second_delays, "F-Score": fscores})
    df.to_csv(os.path.join(artifact_path, f"{follower}_results.csv"), index=False)
    print(f"Names: {np.array(filenames)}")
    print(f"Mean time delay: {np.array(second_delays)}")
    print(f"Mean beat delay: {np.array(beat_delays)}")
    print(f"Mean F-score: {np.array(fscores)}")

