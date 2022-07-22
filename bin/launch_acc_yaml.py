#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import platform
import os

from config_files.brahms_config import accompaniment_match

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import accompanion.accompanist.tempo_models as tempo_models
from accompanion.accompanist import ACCompanion
from accompanion.midi_handler.fluid import FluidsynthPlayer
import mido
import os
import argparse
import glob
from accompanion import PLATFORM

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
    # This creates a RuntimeError: context has already been set.
    if PLATFORM == "Darwin" or PLATFORM == "Linux":
        multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser("Configure and Launch ACCompanion")

    parser.add_argument(
        "--test",
        action="store_true",
        help="switch on Dummy MIDI Router for test environment",
    )

    parser.add_argument("--delay", type=float)
    parser.add_argument(
        "--live",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--bypass_audio",
        default=False,
        help="bypass fluidsynth audio",
        action="store_true",
    )
    parser.add_argument(
        "--use_mediator", default=False, help="use ceus mediator", action="store_true"
    )
    parser.add_argument("--piece")
    parser.add_argument("--follower")
    parser.add_argument(
        "-f", "--config_file", default="test", help="config file to load."
    )
    parser.add_argument("--input", required=False, help="Input MIDI instrument port.")
    parser.add_argument("--output", required=False, help="Output MIDI instrument port.")

    args = parser.parse_args()

    if args.config_file:
        import yaml

        with open(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config_files",
                args.config_file + ".yml",
            ),
            "rb",
        ) as f:
            info_file = yaml.safe_load(f)
        configurations = info_file["config"]
        file_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_pieces",
            info_file["piece_dir"],
        )
        if args.config_file == "simple_pieces":
            configurations["acc_fn"] = os.path.join(file_dir, "secondo.musicxml")
            configurations["solo_fn"] = os.path.join(file_dir, "primo.musicxml")

        else:
            file_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "sample_pieces",
                info_file["piece_dir"],
            )
            configurations["acc_fn"] = os.path.join(
                file_dir, os.path.normpath(info_file["acc_fn"])
            )

            configurations["accompaniment_match"] = os.path.join(
                file_dir, os.path.normpath(info_file["accompaniment_match"])
            ) if "accompaniment_match" in info_file.keys() else None

            configurations["solo_fn"] = glob.glob(
                os.path.join(file_dir, "match", "cc_solo", "*.match")
            )[-5:] if "solo_fn" not in info_file.keys() else os.path.join(
                file_dir, os.path.normpath(info_file["solo_fn"])
            )

            configurations["midi_fn"] = os.path.join(file_dir, os.path.normpath(info_file["midi_fn"])) if "midi_fn" in info_file.keys() else None

    else:
        configurations = dict()

    # import ACCompanion version
    if args.follower:
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
            from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion

            configurations["score_follower_kwargs"] = {
                "score_follower": "OnlineTimeWarping",
                "window_size": 100,
                "step_size": 10,
                "input_processor": {
                    "processor": "PianoRollProcessor",
                    "processor_kwargs": {"piano_range": True},
                },
            }
        else:
            raise ValueError(
                f"console argument 'follower' is of unknown value {args.follower}"
            )
    elif "follower" in configurations.keys():
        if configurations["follower"] == "hmm":
            from accompanion.hmm_accompanion import HMMACCompanion as ACCompanion
        elif configurations["follower"] == "oltw":
            from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion
        else:
            raise ValueError(
                f"configuration parameter 'follower' is of unknown value {configurations['follower']}"
            )
    else:
        raise ValueError(
            "Neither through console arguments nor configuration file has a score follower type been specified"
        )

    if "follower" in configurations.keys():
        del configurations["follower"]

    if args.input:
        configurations["midi_router_kwargs"][
            "solo_input_to_accompaniment_port_name"
        ] = args.input

    if args.output:
        configurations["midi_router_kwargs"][
            "acc_output_to_sound_port_name"
        ] = args.output

    if args.delay is not None:
        configurations["performance_codec_kwargs"]["mechanical_delay"] = args.delay

    if args.piece:
        file_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_pieces",
            args.piece,
        )
        configurations["acc_fn"] = os.path.join(file_dir, "secondo.musicxml")
        configurations["solo_fn"] = os.path.join(file_dir, "primo.musicxml")

    configurations["test"] = True if args.test else False

    accompanion = ACCompanion(**configurations)

    accompanion.run()
    
    

    # try:
    #     accompanion.start()
    # except KeyboardInterrupt:
    #     print("stop_playing")
    #     accompanion.stop_playing()
    #     accompanion.seq.panic_button()
    # finally:
    # try:
    #     accompanion.join()
    # except KeyboardInterrupt:
    #     print("stop_playing")
    #     accompanion.terminate()
    #     # accompanion.stop_playing()
    #     # accompanion.seq.panic_button()
