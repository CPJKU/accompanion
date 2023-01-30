#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import os
import argparse
import yaml
import pandas as pd
import glob
import partitura
from accompanion import PLATFORM
import sys
import numpy as np

# open the file in the write mode

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
sys.path.append("..")
overridable_args = [
    "use_mediator",
    "delay",
    "instrument_port",
    "out_instrument_port",
    "bypass_audio",
    "follower",
    "config_file",
]

if __name__ == "__main__":
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "artifacts")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "artifacts"))
    # if not os.path.exists(os.path.join(os.path.dirname(__file__), "artifacts", "results.csv")):
    #
    #     with open(os.path.join(os.path.dirname(__file__), "artifacts", "results.csv"), 'w') as f:
    #         # create the csv writer
    #         writer = csv.writer(f)
    #         # write a row to the csv file
    #         writer.writerow(["Piece", "Avg Time Delay", "Right Note Ratio", "Played Notes", "Matched Notes", "Score Notes"])
    #

    # This creates a RuntimeError: context has already been set.
    if PLATFORM == "Darwin" or PLATFORM == "Linux":
        multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser("Configure and Launch ACCompanion")

    parser.add_argument(
        "--test",
        action="store_true",
        help="switch on Dummy MIDI Router for test environment",
    )
    parser.add_argument(
        "--bypass_audio",
        default=False,
        help="bypass fluidsynth audio",
        action="store_true",
    )
    parser.add_argument("--follower", default="hmm")
    parser.add_argument("--piece_fn", help="MusicXML file of the primo part or the score")
    parser.add_argument("--acc_fn", help="MusicXML file of the accompaniment part. If None it takes lower staff of piece_fn.", default=None)
    parser.add_argument("--accompaniment_match", help="Match for the accompaniment part (Only for DTW).", default=None)
    parser.add_argument("--midi_fn", help="Midi file to play instead of real time input.")
    parser.add_argument("--window_size", default=100, type=int, help="Window size for the OTW.")
    parser.add_argument("--step_size", default=10, type=int, help="Step size for the OTW.")
    parser.add_argument("--init_bpm", default=120, type=int, help="Initial BPM.")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf_file.yml"), "rb") as f:
        info_file = yaml.safe_load(f)
    configurations = info_file["config"]
    configurations["init_bpm"] = args.init_bpm
    # import ACCompanion version
    if args.follower == "hmm":
        from accompanion.hmm_accompanion import HMMACCompanion as ACCompanion
        configurations["score_follower_kwargs"] = {
            "score_follower": "PitchIOIHMM",
            "input_processor": {
                "processor": "PitchIOIProcessor",
                "processor_kwargs": {"piano_range": True},
            },
        }
    elif args.follower == "oltw":
        # Accompanion Version using random variations of the performance as reference input (created by create_solo_match.py)
        from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion
        match_folder = "match_solo"
        configurations["score_follower_kwargs"] = {
            "score_follower": "OnlineTimeWarping",
            "window_size": args.window_size,
            "step_size": args.step_size,
            "input_processor": {
                "processor": "PianoRollProcessor",
                "processor_kwargs": {"piano_range": True},
            },
        }
    elif args.follower == "oltw_score":
        # Accompanion Version using the score as reference input (created by create_solo_match_score.py)
        from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion
        match_folder = "match_gen"
        configurations["score_follower_kwargs"] = {
            "score_follower": "OnlineTimeWarping",
            "window_size": args.window_size,
            "step_size": args.step_size,
            "input_processor": {
                "processor": "PianoRollProcessor",
                "processor_kwargs": {"piano_range": True},
            },
        }
    elif args.follower == "oltw_test":
        # Accompanion Version using the ground truth match same as the performance.
        from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion

        match_folder = "match_test"
        configurations["score_follower_kwargs"] = {
            "score_follower": "OnlineTimeWarping",
            "window_size": args.window_size,
            "step_size": args.step_size,
            "input_processor": {
                "processor": "PianoRollProcessor",
                "processor_kwargs": {"piano_range": True},
            },
        }
    else:
        raise ValueError(
            f"console argument 'follower' is of unknown value {args.follower}"
        )

    if "follower" in configurations.keys():
        del configurations["follower"]

    if args.piece_fn and os.path.exists(args.piece_fn):
        configurations["acc_fn"] = args.piece_fn
        configurations["solo_fn"] = args.piece_fn
        score_notes = len(partitura.load_score(args.piece_fn).note_array())
    else:
        raise ValueError("No piece file specified or piece path invalid.")

    # Sets Dummy MIDI Router for test environment
    # configurations["test"] = True
    configurations["record_midi"] = False
    if args.midi_fn and os.path.exists(args.midi_fn):
        configurations["midi_fn"] = args.midi_fn
    else:
        raise ValueError("No midi file has been specified, or path MIDI path is invalid.")

    # Load Match files for OLTW
    if args.follower.startswith("oltw"):
        base_name = os.path.splitext(os.path.basename(args.piece_fn))[0]
        match_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.piece_fn)), match_folder, base_name)
        if not os.path.exists(match_dir):
            raise ValueError("No match directory found.")
        configurations["solo_fn"] = glob.glob(
            os.path.join(match_dir, "*.match")
        )

    if configurations["midi_fn"] is not None:

        configurations["midi_router_kwargs"]["solo_input_to_accompaniment_port_name"] = 0
        configurations["midi_router_kwargs"]["MIDIPlayer_to_accompaniment_port_name"] = 0


    accompanion = ACCompanion(**configurations)

    try:
        accompanion.run()
        # Post processing

        performance = partitura.load_performance(args.midi_fn)
        pnote_array = performance.note_array()
        piece_name = os.path.splitext(os.path.basename(args.piece_fn))[0]
        solo_s_onset, solo_p_onset, beat_period = zip(*accompanion.time_delays)
        alignmnent = accompanion.note_tracker.alignment

        for a in alignmnent:
            a["performance_id"] = pnote_array[np.argmin(np.abs(a["onset"] - pnote_array["onset_sec"]))]["id"]

        # for i in range(len(alignmnent)):
        # alignmnent[i]["performance_id"] = alignmnent[i]["onset"]
        #     a = pnote_array[pnote_array["onset_sec"] == alignmnent[i]["onset"]]
        #     if len(a) == 0:
        #         raise ValueError("No note found in performance for onset {}".format(alignmnent[i]["onset"]))
        #     alignmnent["performance_id"] = a["id"].item()

        partitura.io.exportparangonada.save_parangonada_alignment(
            alignmnent,
            os.path.join(os.path.dirname(__file__), "artifacts", f"{piece_name}_{args.follower}_alignment.csv"))

        df = pd.DataFrame({
            "Solo Score Onset": solo_s_onset,
            "Solo Performance Onset": solo_p_onset,
            "Beat Period": beat_period,
        })
        df.to_csv(os.path.join(os.path.dirname(__file__), "artifacts", f"{piece_name}_{args.follower}_time_delays.csv"),
                  index=False)
    # Repeat process in case accompanion is done but processes are stuck (doesn't work on windows).
    except KeyboardInterrupt:
        # Post processing
        performance = partitura.load_performance(args.midi_fn)
        pnote_array = performance.note_array()
        piece_name = os.path.splitext(os.path.basename(args.piece_fn))[0]
        solo_s_onset, solo_p_onset, beat_period = zip(*accompanion.time_delays)
        alignmnent = accompanion.note_tracker.alignment


        # Select the correct performance id for the alignments by using argmin from actual performance array.
        for a in alignmnent:
            a["performance_id"] = pnote_array[np.argmin(np.abs(a["onset"] - pnote_array["onset_sec"]))]["id"]

        partitura.io.exportparangonada.save_parangonada_alignment(
            alignmnent,
            os.path.join(os.path.dirname(__file__), "artifacts", f"{piece_name}_{args.follower}_alignment.csv"))

        df = pd.DataFrame({
            "Solo Score Onset": solo_s_onset,
            "Solo Performance Onset": solo_p_onset,
            "Beat Period": beat_period,
        })
        df.to_csv(os.path.join(os.path.dirname(__file__), "artifacts", f"{piece_name}_{args.follower}_time_delays.csv"),
                  index=False)




