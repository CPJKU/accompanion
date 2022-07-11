#!/usr/bin/env python
# -*- coding: utf-8 -*-
import multiprocessing
import platform
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import accompanion.accompanist.tempo_models as tempo_models
from accompanion.accompanist import ACCompanion
from accompanion.midi_handler.fluid import FluidsynthPlayer


import mido

import os
import argparse
import glob
from accompanion import PLATFORM

overridable_args=['use_mediator','delay','instrument_port','out_instrument_port','bypass_audio','follower','config_file']


# This creates a RuntimeError: context has already been set.
# if PLATFORM == "Darwin" or PLATFORM == "Linux":
# 	multiprocessing.set_start_method("spawn")

parser = argparse.ArgumentParser("Configure and Launch ACCompanion")

parser.add_argument("--delay",type=float)

parser.add_argument("--live", default=False, action="store_true",)
parser.add_argument("--bypass_audio", default=False, help="bypass fluidsynth audio", action="store_true")
parser.add_argument("--use_mediator", default=False, help="use ceus mediator", action="store_true")
parser.add_argument("--piece", default="twinkle_twinkle_little_star")
parser.add_argument("--follower")
parser.add_argument("-f", "--config_file", default="brahms", help="config file to load.")
parser.add_argument('--input', required=False, help="Input MIDI instrument port.")
parser.add_argument('--output', required=False, help="Output MIDI instrument port.")

args = parser.parse_args()

if args.config_file:
	import yaml
	with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config_files", args.config_file + ".yml"), "rb") as f:
		info_file = yaml.safe_load(f)
	configurations = info_file["config"]
	file_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_pieces", info_file["piece_dir"])
	configurations["acc_fn"] = os.path.join(file_dir, os.path.normpath(info_file["acc_fn"]))
	configurations["solo_fn"] = glob.glob(os.path.join(file_dir, "match", "cc_solo", "*.match"))[-5:]
	configurations["accompaniment_match"] = os.path.join(file_dir, os.path.normpath(info_file["accompaniment_match"]))
	# configurations["midi_fn"] = os.path.join(file_dir, os.path.normpath(info_file["midi_fn"]))
else:
	configurations = dict()


#import ACCompanion version
if args.follower:
	if args.follower == "hmm":
		from accompanion.hmm_accompanion import HMMACCompanion as ACCompanion
		configurations['score_follower_kwargs']={
			"score_follower": "PitchIOIHMM",
			"input_processor": {
				"processor": "PitchIOIProcessor",
				"processor_kwargs": {"piano_range": True},
			},
		}
	elif args.follower == "oltw":
		from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion
		configurations['score_follower_kwargs']={
			"score_follower": "OnlineTimeWarping",
			"window_size": 100,
			"step_size": 10,
			"input_processor": {
				"processor": "PianoRollProcessor",
				"processor_kwargs": {"piano_range": True},
			},
		}
	else:
		raise ValueError(f"console argument 'follower' is of unknown value {args.follower}")
elif 'follower' in configurations.keys():
	if configurations['follower']=='hmm':
		from accompanion.hmm_accompanion import HMMACCompanion as ACCompanion
	elif configurations['follower']=='oltw':
		from accompanion.oltw_accompanion import OLTWACCompanion as ACCompanion
	else:
		raise ValueError(f"configuration parameter 'follower' is of unknown value {configurations['follower']}")
else:
	raise ValueError('Neither through console arguments or configuration file has a score follower type been specified')


if 'follower' in configurations.keys():
	del configurations['follower']



if args.input:
	configurations['midi_router_kwargs']['solo_input_to_accompaniment_port_name'] = args.input

if args.output:
	configurations['midi_router_kwargs']['acc_output_to_sound_port_name'] = args.output

if args.delay:
	configurations['performance_codec_kwargs']['mechanical_delay'] = args.delay



accompanion = ACCompanion(**configurations)

try:
	accompanion.start()
except KeyboardInterrupt:
	print("stop_playing")
	accompanion.stop_playing()
	accompanion.seq.panic_button()
finally:
	accompanion.join()