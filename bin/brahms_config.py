from accompanion.accompanist.tempo_models import LTESM as tempo_model
import os
import glob
from accompanion.midi_handler.fluid import FluidsynthPlayer

file_dir = os.path.dirname(os.path.abspath(__file__))
rel_path_from_CWD = os.path.relpath(file_dir, os.curdir)

brahms_dir = os.path.join(rel_path_from_CWD, "..", "sample_pieces", "brahms_data")
acc_fn = os.path.join(
    brahms_dir, "musicxml", "Brahms_Hungarian-Dance-5_Secondo.musicxml"
)
solo_fn = glob.glob(os.path.join(brahms_dir, "match", "cc_solo", "*.match"))[-5:]
midi_fn = os.path.join(
    brahms_dir,
    "midi",
    "cc_solo",
    "Brahms_Hungarian-Dance-5_Primo_2021-07-27.mid",
)
accompaniment_match = os.path.join(
    brahms_dir, "basismixer", "bm_brahms_2021-08-30.match"
)

config = dict(
	acc_fn = acc_fn,
	solo_fn = solo_fn,
	midi_fn = midi_fn,
	accompaniment_match = accompaniment_match,
	follower = "oltw",
	score_follower_kwargs={
        "score_follower": "OnlineTimeWarping",
        "window_size": 100,
        "step_size": 10,
        "input_processor": {
            "processor": "PianoRollProcessor",
            "processor_kwargs": {"piano_range": True},
        },
    },
    performance_codec_kwargs = {
        "velocity_trend_ma_alpha": 0.6,
        "articulation_ma_alpha": 0.4,
        "velocity_dev_scale": 70,
        "velocity_min": 20,
        "velocity_max": 100,
        "velocity_solo_scale": 0.85,
        "timing_scale": 0.001,
        "log_articulation_scale": 0.1,
        "mechanical_delay": 0.210,
    },
    midi_router_kwargs = dict(
        solo_input_to_accompaniment_port_name='MPKmini2:MPKmini2 MIDI 1 24:0',
        acc_output_to_sound_port_name=FluidsynthPlayer,
        MIDIPlayer_to_sound_port_name=FluidsynthPlayer,
        MIDIPlayer_to_accompaniment_port_name=0,
        simple_button_input_port_name=None,
    ),
    adjust_following_rate=0.2,
    bypass_audio=False,
    tempo_model_kwargs={"tempo_model": tempo_model},
    use_ceus_mediator=False,
    polling_period=0.01,
    init_bpm = 120,
    init_velocity = 60
)