# !/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import glob
import multiprocessing

# import tkinter as tk
# import mido
# from copy import deepcopy
# import time
import os
import sys

import config_gui

from accompanion import PLATFORM

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

    # This creates a RuntimeError: context has already been set.
    if PLATFORM == "Darwin" or PLATFORM == "Linux":
        multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser("Configure and Launch ACCompanion")

    parser.add_argument(
        "--skip_gui", action="store_true", help="skip configuration gui at startup"
    )

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
    parser.add_argument(
        "--record-midi", action="store_true", help="Record Midi input and Output."
    )
    parser.add_argument(
        "--midi-fn", help="Midi file to play instead of real time input."
    )

    args = parser.parse_args()

    if not args.skip_gui:
        (
            configurations,
            ACCompanion,
        ) = config_gui.accompanion_configurations_and_version_via_gui()

        if configurations is None:
            import sys

            sys.exit()

        if "midi_fn" in configurations.keys() and configurations["midi_fn"] == "":
            configurations["midi_fn"] = None
    elif args.config_file:
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
        if args.config_file in ["brahms", "mozart", "schubert", "fourhands", "FH"]:
            args.follower = "oltw"
            file_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "accompanion_pieces",
                "complex_pieces",
                info_file["piece_dir"],
            )
            configurations["acc_fn"] = os.path.join(
                file_dir, os.path.normpath(info_file["acc_fn"])
            )
            configurations["accompaniment_match"] = (
                os.path.join(
                    file_dir, os.path.normpath(info_file["accompaniment_match"])
                )
                if "accompaniment_match" in info_file.keys()
                else None
            )
            configurations["solo_fn"] = (
                glob.glob(os.path.join(file_dir, "match", "cc_solo", "*.match"))[-5:]
                if "solo_fn" not in info_file.keys()
                else os.path.join(file_dir, os.path.normpath(info_file["solo_fn"]))
            )
            configurations["midi_fn"] = (
                os.path.join(file_dir, os.path.normpath(info_file["midi_fn"]))
                if "midi_fn" in info_file.keys()
                else None
            )
        else:
            file_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "accompanion_pieces",
                "simple_pieces",
                info_file["piece_dir"],
            )
            # If only one piece is available load and separate to parts
            if os.path.join(file_dir, "primo.musicxml") not in glob.glob(
                os.path.join(file_dir, "*.musicxml")
            ):
                import partitura as pt
                score = pt.load_score(
                    (
                        glob.glob(os.path.join(file_dir, "*.musicxml"))
                        + glob.glob(os.path.join(file_dir, "*.mxl"))
                    )[0]
                )
                if len(score.parts) == 1:
                    from copy import deepcopy

                    import numpy as np

                    # find individual staff
                    na = score.note_array(include_staff=True)
                    staff = np.unique(na["staff"])
                    primo_part = deepcopy(score.parts[0])
                    secondo_part = deepcopy(score.parts[0])
                    for st in staff:
                        if st == 1:
                            primo_part.notes = [
                                note for note in primo_part.notes if note.staff == st
                            ]
                        else:
                            secondo_part.notes = [
                                note for note in secondo_part.notes if note.staff == st
                            ]

                elif len(score.parts) == 2:
                    primo_part = score.parts[0]
                    pt.save_musicxml(
                        primo_part, os.path.join(file_dir, "primo.musicxml")
                    )
                    secondo_part = score.parts[1]
                    pt.save_musicxml(
                        secondo_part, os.path.join(file_dir, "secondo.musicxml")
                    )
                else:
                    raise ValueError("Score has more than two parts.")

            configurations["acc_fn"] = os.path.join(file_dir, "secondo.musicxml")
            configurations["solo_fn"] = os.path.join(file_dir, "primo.musicxml")

    else:
        configurations = dict()

    # import ACCompanion version if not already done so by GUI
    if not args.skip_gui:
        pass
    elif args.follower:
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
    configurations["record_midi"] = args.record_midi if args.record_midi else False
    configurations["midi_fn"] = args.midi_fn if args.midi_fn else None

    if configurations["midi_fn"] is not None:
        configurations["midi_router_kwargs"][
            "solo_input_to_accompaniment_port_name"
        ] = 0
        configurations["midi_router_kwargs"][
            "MIDIPlayer_to_accompaniment_port_name"
        ] = 0

    accompanion = ACCompanion(**configurations)

    accompanion.run()


# This file will be used to create a GUI for the ACCompanion
# The GUI will be used to select the input and output ports available
# and to start the accompaniment by running the script launch_acc.py with the
# selected ports as arguments and the selected accompaniment type as argument
# The GUI will also be used to stop the accompaniment by running the script by pushing the button stop that performs a kill command on the process

# The GUI will be created using tkinter

# The GUI will be composed of a window with a title and a frame with a title

# The frame will contain a list of available input ports and a list of available output ports

# class ACCompanionApp:
#     global is_loading
#     def __init__(self):
#
#         self.root = tk.Tk()
#         self.root.title("ACCompanion")
#         self.root.attributes("-fullscreen", True)
#         self.root.configure(background="black")
#         self.init_frame()
#         self.root.mainloop()
#
#     def init_frame(self):
#         # create the frame
#         # Fullscreen
#         self.frame = tk.Frame(self.root, width=800, height=480)
#         self.frame.configure(background="black")
#
#         self.input_ports = mido.get_input_names()
#         self.output_ports = mido.get_output_names()
#
#         # The user can select the input port and the output port from the list:
#         self.input_port = tk.StringVar(self.frame)
#         self.input_port.set("Select MIDI Input Port")  # default value
#         self.output_port = tk.StringVar(self.frame)
#         self.output_port.set("Select MIDI Output Port")  # default value
#         file_dir = os.path.join(
#             os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#             "accompanion_pieces",
#             "simple_pieces")
#         self.available_pieces = sorted(os.listdir(file_dir))
#         self.available_pieces_title = tk.StringVar(self.frame)
#         self.available_pieces_title.set("Select Accompaniment Piece")  # default value
#
#         # create the list of input ports
#         self.input_ports_list = tk.OptionMenu(self.frame, self.input_port, *self.input_ports)
#         self.input_ports_list.pack(side="left")
#
#
#         # create the list of output ports
#         self.output_ports_list = tk.OptionMenu(self.frame, self.output_port, *self.output_ports)
#         self.output_ports_list.pack(side="left")
#
#         # create the list of available pieces
#         self.available_pieces_list = tk.OptionMenu(self.frame, self.available_pieces_title, *self.available_pieces)
#         self.available_pieces_list.pack(side="left")
#
#         # create the button to start the accompaniment
#         self.start_button = tk.Button(self.frame, text="Start", command=self.start_accompaniment)
#         self.start_button.pack(side="left")
#
#         # Run the app
#         self.frame.pack()
#
#
#     def start_accompaniment(self):
#         # Runs self.load_accompanion as a process and shows the loading screen
#         # self.load_accompanion()
#
#
#         q = multiprocessing.Queue()
#         acc_load = multiprocessing.Process(target=self.load_accompanion, args=(q,))
#         acc_load.start()
#         self.loading_sceen(q)
#
#         # Create the frame for the accompaniment
#         self.accompaniment_frame = tk.Frame(self.root)
#         # Create the button to stop the accompaniment
#         self.stop_button = tk.Button(self.accompaniment_frame, text="Stop", command=lambda: self.stop_accompaniment(q))
#         self.stop_button.pack(side="left")
#
#         # Project text that shows Now Playing
#
#         tk.Label(self.accompaniment_frame, text="Now Playing:", bg="black", fg="#FFBD09",
#                  font="Bahnschrift 15", ).place(x=490, y=320)
#         self.accompaniment_frame.pack()
#
#         self.accompaniment_frame.tkraise()
#         self.loading_frame.destroy()
#
#
#     def loading_sceen(self, output_queue):
#
#         self.loading_frame = tk.Frame(self.root, width=800, height=480)
#         self.loading_frame.configure(background="black")
#
#         # loading text:
#         tk.Label(self.loading_frame, text="Loading...", bg="black", fg="#FFBD09", font="Bahnschrift 15", ).place(x=490, y=320)
#
#         # loading blocks:
#         for i in range(16):
#             tk.Label(self.loading_frame, bg="#1F2732", width=2, height=1).place(x=(i + 22) * 22, y=350)
#
#         self.loading_frame.pack()
#         self.loading_frame.tkraise()
#         self.frame.destroy()
#         self.loading_frame.update()
#
#         while True:
#             if not output_queue.empty():
#                 if output_queue.get() == "done":
#                     break
#             self.play_loading()
#
#     def play_loading(self):
#         for i in range(200):
#             for j in range(16):
#                 # make block yellow:
#                 tk.Label(self.loading_frame, bg="#FFBD09", width=2, height=1).place(x=(j + 22) * 22, y=350)
#                 time.sleep(0.1)
#                 self.loading_frame.update_idletasks()
#                 # make block dark:
#                 tk.Label(self.loading_frame, bg="#1F2732", width=2, height=1).place(x=(j + 22) * 22, y=350)
#
#     def stop_accompaniment(self, output_queue):
#         # Kills the process running the accompaniment
#         self.root.destroy()
#         raise KeyboardInterrupt
#
#     def load_accompanion(self, output_queue):
#         # This creates a RuntimeError: context has already been set.
#         # if PLATFORM == "Darwin" or PLATFORM == "Linux":
#         #     multiprocessing.set_start_method("spawn")
#
#         parser = argparse.ArgumentParser("Configure and Launch ACCompanion")
#
#         parser.add_argument(
#             "--test",
#             action="store_true",
#             help="switch on Dummy MIDI Router for test environment",
#         )
#
#         parser.add_argument("--delay", type=float)
#         parser.add_argument(
#             "--live",
#             default=False,
#             action="store_true",
#         )
#         parser.add_argument(
#             "--bypass_audio",
#             default=False,
#             help="bypass fluidsynth audio",
#             action="store_true",
#         )
#         parser.add_argument(
#             "--use_mediator", default=False, help="use ceus mediator", action="store_true"
#         )
#         parser.add_argument("--piece")
#         parser.add_argument("--follower", default="hmm")
#         parser.add_argument(
#             "-f", "--config_file", default="test", help="config file to load."
#         )
#         parser.add_argument("--input", required=False, help="Input MIDI instrument port.")
#         parser.add_argument("--output", required=False, help="Output MIDI instrument port.")
#         parser.add_argument("--record-midi", action="store_true", help="Record Midi input and Output.")
#         parser.add_argument("--midi-fn", help="Midi file to play instead of real time input.")
#
#         args = parser.parse_args()
#
#         args.input = self.input_port.get()
#         args.output = self.output_port.get()
#         args.piece = self.available_pieces_title.get()
#
#         if args.config_file:
#             import yaml
#
#             with open(
#                     os.path.join(
#                         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#                         "config_files",
#                         args.config_file + ".yml",
#                     ),
#                     "rb",
#             ) as f:
#                 info_file = yaml.safe_load(f)
#             configurations = info_file["config"]
#             # TODO : add a configuration for the default loaded file and directories.
#             if args.config_file in ["brahms", "mozart", "schubert", "fourhands", "FH"]:
#                 args.follower = "oltw"
#                 file_dir = os.path.join(
#                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#                     "accompanion_pieces",
#                     "complex_pieces",
#                     info_file["piece_dir"],
#                 )
#                 configurations["acc_fn"] = os.path.join(file_dir, os.path.normpath(info_file["acc_fn"]))
#                 configurations["accompaniment_match"] = os.path.join(
#                     file_dir, os.path.normpath(info_file["accompaniment_match"])
#                 ) if "accompaniment_match" in info_file.keys() else None
#                 configurations["solo_fn"] = glob.glob(
#                     os.path.join(file_dir, "match", "cc_solo", "*.match")
#                 )[-5:] if "solo_fn" not in info_file.keys() else os.path.join(
#                     file_dir, os.path.normpath(info_file["solo_fn"])
#                 )
#                 configurations["midi_fn"] = os.path.join(file_dir, os.path.normpath(
#                     info_file["midi_fn"])) if "midi_fn" in info_file.keys() else None
#             else:
#                 file_dir = os.path.join(
#                     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#                     "accompanion_pieces",
#                     "simple_pieces",
#                     args.piece
#                 )
#                 # If only one piece is available load and separate to parts
#                 if os.path.join(file_dir, "primo.musicxml") not in glob.glob(os.path.join(file_dir, "*.musicxml")):
#                     import partitura as pt#
#                     score pt.load_score(
#                         (glob.glob(os.path.join(file_dir, "*.musicxml")) + glob.glob(os.path.join(file_dir, "*.mxl")))[
#                             0])
#                     if len(score.parts) == 1:
#                         import numpy as np
#
#                         # find individual staff
#                         na = score.note_array(include_staff=True)
#                         staff = np.unique(na["staff"])
#                         primo_part = deepcopy(score.parts[0])
#                         secondo_part = deepcopy(score.parts[0])
#                         for st in staff:
#                             if st == 1:
#                                 primo_part.notes = [note for note in primo_part.notes if note.staff == st]
#                             else:
#                                 secondo_part.notes = [note for note in secondo_part.notes if note.staff == st]
#
#                     elif len(score.parts) == 2:
#                         primo_part = score.parts[0]
#                         partitura.save_musicxml(primo_part, os.path.join(file_dir, "primo.musicxml"))
#                         secondo_part = score.parts[1]
#                         partitura.save_musicxml(secondo_part, os.path.join(file_dir, "secondo.musicxml"))
#                     else:
#                         raise ValueError("Score has more than two parts.")
#
#                 configurations["acc_fn"] = os.path.join(file_dir, "secondo.musicxml")
#                 configurations["solo_fn"] = os.path.join(file_dir, "primo.musicxml")
#
#         else:
#             configurations = dict()
#
#         # import ACCompanion version
#         if args.follower:
#             if args.follower == "hmm":
#                 from accompanion.hmm_accompanion import HMMACCompanion as ACCompanion
#
#                 configurations["score_follower_kwargs"] = {
#                     "score_follower": "PitchIOIHMM",
#                     "input_processor": {
#                         "processor": "PitchIOIProcessor",
#                         "processor_kwargs": {"piano_range": True},
#                     },
#                 }
#             elif args.follower == "oltw":
#                 from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion
#
#                 configurations["score_follower_kwargs"] = {
#                     "score_follower": "OnlineTimeWarping",
#                     "window_size": 100,
#                     "step_size": 10,
#                     "input_processor": {
#                         "processor": "PianoRollProcessor",
#                         "processor_kwargs": {"piano_range": True},
#                     },
#                 }
#             else:
#                 raise ValueError(
#                     f"console argument 'follower' is of unknown value {args.follower}"
#                 )
#         elif "follower" in configurations.keys():
#             if configurations["follower"] == "hmm":
#                 from accompanion.hmm_accompanion import HMMACCompanion as ACCompanion
#             elif configurations["follower"] == "oltw":
#                 from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion
#             else:
#                 raise ValueError(
#                     f"configuration parameter 'follower' is of unknown value {configurations['follower']}"
#                 )
#         else:
#             raise ValueError(
#                 "Neither through console arguments nor configuration file has a score follower type been specified"
#             )
#
#         if "follower" in configurations.keys():
#             del configurations["follower"]
#
#         if args.input:
#             configurations["midi_router_kwargs"][
#                 "solo_input_to_accompaniment_port_name"
#             ] = args.input
#
#         if args.output:
#             configurations["midi_router_kwargs"][
#                 "acc_output_to_sound_port_name"
#             ] = args.output
#
#         if args.delay is not None:
#             configurations["performance_codec_kwargs"]["mechanical_delay"] = args.delay
#
#         if args.piece:
#             file_dir = os.path.join(
#                 os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#                 "accompanion_pieces",
#                 "simple_pieces",
#                 args.piece,
#             )
#             configurations["acc_fn"] = os.path.join(file_dir, "secondo.musicxml")
#             configurations["solo_fn"] = os.path.join(file_dir, "primo.musicxml")
#
#         configurations["test"] = True if args.test else False
#
#         configurations["record_midi"] = args.record_midi if args.record_midi else False
#         configurations["midi_fn"] = args.midi_fn if args.midi_fn else None
#
#         if configurations["midi_fn"] is not None:
#             configurations["midi_router_kwargs"]["solo_input_to_accompaniment_port_name"] = 0
#             configurations["midi_router_kwargs"]["MIDIPlayer_to_accompaniment_port_name"] = 0
#
#
#         # self.run_accompanion()
#         output_queue.put("done")
#
#         accompanion = ACCompanion(**configurations)
#         accompanion.run()
#
#
#
# if __name__ == "__main__":
#     # launch the app
#     app = ACCompanionApp()
