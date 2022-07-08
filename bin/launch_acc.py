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
import signal

overridable_args=['use_mediator','delay','instrument_port','out_instrument_port','bypass_audio','follower','config_file']


if __name__ == "__main__":

	if PLATFORM == "Darwin" or PLATFORM == "Linux":
	    multiprocessing.set_start_method("spawn")

	parser = argparse.ArgumentParser("Configure and Launch ACCompanion")

	parser.add_argument("--delay",type=float)

	parser.add_argument("--live", default=False, action="store_true",)

	parser.add_argument(
	    "--bypass_audio",
	    default=False,
	    help="bypass fluidsynth audio",
	    action="store_true",
	)

	parser.add_argument(
	    "--use_mediator", default=False, help="use ceus mediator", action="store_true",
	)

	#parser.add_argument("--piece", default="twinkle_twinkle_little_star")

	parser.add_argument("--follower")

	parser.add_argument("--config_file")

	parser.add_argument('--instrument_port')

	parser.add_argument('--out_instrument_port')

	args = parser.parse_args()

	if args.config_file:
		configurations = getattr(__import__(args.config_file),'config')

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


	if args.instrument_port:
		configurations['midi_router_kwargs']['solo_input_to_accompaniment_port_name'] = args.instrument_port

	if args.out_instrument_port:
		configurations['midi_router_kwargs']['acc_output_to_sound_port_name']=args.out_instrument_port

	if args.delay:
		configurations['performance_codec_kwargs']['mechanical_delay']=args.delay



	accompanion = ACCompanion(**configurations)

	try:
	    accompanion.start()
	except KeyboardInterrupt:
	    print("stop_playing")
	    accompanion.stop_playing()
	    accompanion.seq.panic_button()
	finally:
	    accompanion.join()