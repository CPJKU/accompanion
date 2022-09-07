from accompanion.accompanist.tempo_models import LTESM as tempo_model
import os
import glob
from accompanion.midi_handler.fluid import FluidsynthPlayer

file_dir = os.path.dirname(os.path.abspath(__file__))
rel_path_from_CWD = os.path.relpath(file_dir, os.curdir)

mozart_dir = os.path.join(rel_path_from_CWD, "..", "sample_pieces", "mozart_demo")
acc_fn = os.path.join(
    mozart_dir, "musicxml", "Sonata-k381-a123_Secondo.musicxml"
)
solo_fn = glob.glob(os.path.join(mozart_dir, "match", "primo", "*.match"))[-6:]
midi_fn = os.path.join(
    mozart_dir,
    "midi",
    "primo",
    "Sonata-k381-a123_Primo_v1.mid",
)
accompaniment_match = os.path.join(
    mozart_dir, "basismixer", "mozart_sonata_secondo.match"
)

config = dict(
	acc_fn = acc_fn,
	solo_fn = solo_fn,
	midi_fn = None,
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
        solo_input_to_accompaniment_port_name='Clavinova',
        acc_output_to_sound_port_name='Clavinova',
        MIDIPlayer_to_sound_port_name=None,
        MIDIPlayer_to_accompaniment_port_name=None,
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