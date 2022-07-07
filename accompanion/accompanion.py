# -*- coding: utf-8 -*-
"""
ACCompanion!

TODO
----
* Add visualization stuff!
"""
import multiprocessing
import threading
import time

import numpy as np
import partitura

from typing import Optional
from basismixer.performance_codec import get_performance_codec
from basismixer.utils.music import onsetwise_to_notewise, notewise_to_onsetwise
from scipy.interpolate import interp1d

from accompanion.midi_handler.midi_input import create_midi_poll, POLLING_PERIOD
from accompanion.midi_handler.midi_file_player import get_midi_file_player
from accompanion.midi_handler.midi_sequencing_threads import ScoreSequencer
from accompanion.midi_handler.midi_routing import MidiRouter

# from accompanion.mtchmkr.features_midi import PianoRollProcessor
# from accompanion.mtchmkr.alignment_online_oltw import OnlineTimeWarping
# from accompanion.mtchmkr.score_hmm import PitchIOIHMM


from accompanion.mtchmkr.utils_generic import SequentialOutputProcessor

from accompanion.accompanist.score import (
    part_to_score,
    alignment_to_score,
    AccompanimentScore,
    Score
)

from accompanion.accompanist.accompaniment_decoder import (
    OnlinePerformanceCodec,
    Accompanist,
    moving_average_offline,
)
import accompanion.accompanist.tempo_models as tempo_models

from accompanion.utils.partitura_utils import (
    get_time_maps_from_alignment,
    partitura_to_framed_midi_custom as partitura_to_framed_midi,
    # get_beat_conversion,
    DECAY_VALUE,
)

from accompanion.midi_handler.ceus_mediator import CeusMediator
from accompanion.score_follower.note_tracker import NoteTracker
from accompanion.score_follower.onset_tracker import OnsetTracker
from accompanion.score_follower.trackers import MultiDTWScoreFollower, AccompanimentScoreFollower
from accompanion.midi_handler.fluid import FluidsynthPlayer


ACC_PROCESS = True
ACC_PARENT = multiprocessing.Process if ACC_PROCESS else threading.Thread
USE_THREADS = True


class ACCompanion(ACC_PARENT):
    def __init__(
        self,
        solo_score: Score,
        accompaniment_score: AccompanimentScore,
        score_follower: AccompanimentScoreFollower,
        tempo_model: tempo_models.SyncModel,
        performance_codec: OnlinePerformanceCodec,
        input_pipeline: SequentialOutputProcessor,
        midi_router: MidiRouter,
        midi_fn: Optional[str] = None,
        init_bpm: float = 60,
        init_velocity: int = 60,
        polling_period: bool = POLLING_PERIOD,
        use_ceus_mediator: bool = False,
        adjust_following_rate: float = 0.1,
        bypass_audio: bool = False,  # bypass fluidsynth audio
    ) -> None:
        super(ACCompanion, self).__init__()

        self.solo_score = solo_score
        self.acc_score = accompaniment_score
        self.midi_fn = midi_fn

        self.router = midi_router

        self.init_bpm = init_bpm
        self.init_bp = 60 / self.init_bpm
        self.init_velocity = init_velocity

        # Parameters for following
        self.polling_period = polling_period

        self.follower = score_follower

        self.tempo_model = tempo_model

        self.bypass_audio = bypass_audio

        self.first_score_onset = None
        self.adjust_following_rate = adjust_following_rate
        # Rate in "loops_without_update"  for adjusting the score
        # follower with expected position at the
        # current tempo

        self.accompanist = Accompanist(
            accompaniment_score=self.acc_score,
            performance_codec=performance_codec,
        )

        if self.use_mediator:
            self.mediator = CeusMediator()

        self.seq = ScoreSequencer(
            score_or_notes=self.acc_score,
            outport=self.router.acc_output_to_sound_port,
            mediator=self.mediator,
        )
        self.seq.panic_button()

        # Update tempo model
        self.tempo_model.beat_period = self.init_bp
        self.prev_score_onset = self.solo_score.unique_onsets.min()
        self.first_score_onset = self.solo_score.unique_onsets.min()

        # Update tempo model
        self.tempo_model.beat_period = self.init_bp
        self.prev_score_onset = self.solo_score.unique_onsets.min()
        self.first_score_onset = self.solo_score.unique_onsets.min()

        # initialize note tracker
        self.note_tracker = NoteTracker(self.solo_spart.note_array())
        self.accompanist.pc.note_tracker = self.note_tracker

        self.pipe_out, self.queue, self.midi_input_process = create_midi_poll(
            port_name=self.router.solo_input_to_accompaniment_port_name[1],
            polling_period=self.polling_period,
            # velocities only for visualization purposes
            pipeline=input_pipeline,
            # pipeline=SequentialOutputProcessor(
            #     [PianoRollProcessor(piano_range=True, use_velocity=True)]
            # ),
            return_midi_messages=True,
            thread=USE_THREADS,
            mediator=self.mediator,
        )

    @property
    def beat_period(self):
        return self.beat_period_

    @beat_period.setter
    def beat_period(self, beat_period):
        self.beat_period_ = beat_period

        if self.accompanist is not None:
            self.accompanist.pc.bp_ave = beat_period

    @property
    def velocity(self):
        return self.velocity_

    @velocity.setter
    def velocity(self, velocity):
        self.velocity_ = velocity

    def stop_playing(self):
        self.play_accompanion = False

        if self.dummy_solo is not None:
            self.dummy_solo.stop_playing()

        self.midi_input_process.stop_listening()
        self.seq.stop_playing()
        self.router.close_ports()

    def terminate(self):

        if hasattr(super(ACCompanion, self), "terminate"):
            super(ACCompanion, self).terminate()
        else:
            self.stop_playing()

    def run(self):

        self.play_accompanion = True
        solo_starts = True
        sequencer_start = False
        start_time = None

        # start the accompaniment if the solo part starts afterwards
        if self.acc_score.min_onset < self.solo_score.min_onset:
            self.seq.init_time = time.time()
            self.accompanist.pc.prev_eq_onset = 0
            self.seq.start()
            sequencer_start = True
            solo_starts = False
            start_time = self.seq.init_time

        # intialize beat period
        # perf_start = False

        onset_tracker = OnsetTracker(self.solo_score.unique_onsets)
        # Initialize on-line Basis Mixer here
        # expression_model = BasisMixer()
        self.midi_input_process.start()
        print("Start listening")

        self.perf_frame = None
        self.score_idx = 0

        if self.midi_fn is not None:
            print("Start playing MIDI file")
            self.dummy_solo = get_midi_file_player(
                port_name=self.router.MIDIPlayer_to_accompaniment_port_name[1],
                file_name=self.midi_fn,
                player_class=FluidsynthPlayer,
                thread=USE_THREADS,
                bypass_audio=self.bypass_audio,
            )
            self.dummy_solo.start()

        # dummy start time (see below)
        if start_time is None:
            start_time = time.time()
            if not sequencer_start:
                self.seq.init_time = start_time
        expected_position = self.first_score_onset
        loops_without_update = 0
        empty_loops = 0
        prev_solo_p_onset = None
        adjusted_sf = False
        decay = np.ones(88)

        pioi = self.polling_period

        try:
            while self.play_accompanion and not self.seq.end_of_piece:

                if self.queue.poll():
                    output = self.queue.recv()
                    # CC: moved solo_p_onset here because of the delays...
                    # perhaps it would be good to take the time from
                    # the MIDI messages?
                    solo_p_onset = time.time() - start_time

                    input_midi_messages, output = output

                    # listen to metronome notes for tempo
                    # copy output to perf_frame
                    # (with velocities for visualization)
                    self.perf_frame = output.copy()
                    # overwrite velocities to 1 for tracking
                    # TODO think about nicer solution
                    # output[output > 0] = 1.0

                    # # start playing the performance
                    # if not perf_start and (output > 0).any():
                    #     # Ignore messages after the tapping
                    #     if np.all(
                    #         np.where(output > 0)[0] + 21
                    #         == self.solo_score.getitem_indexwise(0).pitch
                    #     ):
                    #         perf_start = True
                    #         print("start following!")

                    # Use these onset times?
                    onset_times = [
                        msg[1]
                        for msg in input_midi_messages
                        if msg[0].type in ("note_on", "note_off")
                    ]
                    onset_time = np.mean(onset_times) if len(onset_times) > 0 else 0
                    new_midi_messages = False
                    decay *= DECAY_VALUE
                    for msg, msg_time in input_midi_messages:
                        if msg.type in ("note_on", "note_off"):

                            if msg.type == "note_on" and msg.velocity > 0:
                                new_midi_messages = True
                            midi_msg = (msg.type, msg.note, msg.velocity, onset_time)
                            self.note_tracker.track_note(midi_msg)

                            decay[msg.note - 21] = 1.0

                    output *= decay

                    if sum(output) == 0:
                        empty_loops += 1
                    else:
                        empty_loops == 0

                    # if perf_start:
                    self.score_idx, score_position = self.score_follower(output)
                    solo_s_onset, onset_index, acc_update = onset_tracker(
                        score_position,
                        expected_position
                        # self.seq.performed_score_onsets[-1]
                    )

                    pioi = (
                        solo_p_onset - prev_solo_p_onset
                        if prev_solo_p_onset is not None
                        else self.polling_period
                    )
                    prev_solo_p_onset = solo_p_onset
                    expected_position = expected_position + pioi / self.beat_period

                    if solo_s_onset is not None:

                        print(
                            f"performed onset {solo_s_onset}",
                            f"expected onset {expected_position}",
                            f"beat_period {self.beat_period}",
                            f"adjusted {acc_update or adjusted_sf}",
                        )

                        if not acc_update:2 
                            asynch = expected_position - solo_s_onset
                            # print('asynchrony', asynch)
                            expected_position = expected_position - 0.6 * asynch
                            loops_without_update = 0
                            adjusted_sf = False
                        else:
                            loops_without_update += 1

                        if new_midi_messages:
                            self.note_tracker.update_alignment(solo_s_onset)
                        # start accompaniment if it starts at the
                        # same time as the solo
                        if solo_starts and onset_index == 0:
                            if not sequencer_start:
                                print("Start accompaniment")
                                sequencer_start = True
                                self.accompanist.accompaniment_step(
                                    solo_s_onset=solo_s_onset,
                                    solo_p_onset=solo_p_onset,
                                )
                                self.seq.start()

                        if (
                            solo_s_onset > self.first_score_onset
                            and not acc_update
                            and not adjusted_sf
                        ):
                            self.accompanist.accompaniment_step(
                                solo_s_onset=solo_s_onset, solo_p_onset=solo_p_onset
                            )
                            self.beat_period = self.accompanist.pc.bp_ave
                    else:
                        loops_without_update += 1

                    if loops_without_update % self.afr == 0:
                        # only allow forward updates
                        if self.score_follower.current_position < expected_position:
                            self.score_follower.update_position(expected_position)
                            adjusted_sf = True

                    if self.pipe is not None:
                        self.pipe.send(
                            (
                                self.perf_frame,
                                self.get_accompaniment_frame(),
                                self.score_idx,
                                self.get_tempo(),
                            )
                        )

        except Exception:
            pass
        finally:
            self.stop_playing()


if __name__ == "__main__":

    import os
    import argparse
    import glob

    from accompanion import PLATFORM

    if PLATFORM == "Darwin" or PLATFORM == "Linux":
        multiprocessing.set_start_method("spawn")

    file_dir = os.path.dirname(os.path.abspath(__file__))
    rel_path_from_CWD = os.path.relpath(file_dir, os.curdir)

    piece = "brahms"

    if piece == "brahms":
        brahms_dir = os.path.join(
            rel_path_from_CWD, "..", "sample_pieces", "brahms_data"
        )
        acc_fn = os.path.join(
            brahms_dir, "musicxml", "Brahms_Hungarian-Dance-5_Secondo.musicxml"
        )
        solo_fn = glob.glob(os.path.join(brahms_dir, "match", "cc_solo", "*.match"))[
            -5:
        ]
        midi_fn = os.path.join(
            brahms_dir,
            "midi",
            "cc_solo",
            "Brahms_Hungarian-Dance-5_Primo_2021-07-27.mid"
        )
        accompaniment_match = os.path.join(
            brahms_dir, "basismixer", "bm_brahms_2021-08-30.match"
        )

        tempo_tapping = (4, "quarter")
        init_bpm = 120

        tempo_model = tempo_models.LTESM

    parser = argparse.ArgumentParser("ACCompanion")

    # TODO: Using all 15 performances makes the following
    # too slow (we have to call the follower 15 times).
    # Unless we find a way to paralellize multiple followers
    # We need to check what is the "ideal" number, and which
    # performances would work better.
    parser.add_argument(
        "--solo-fn",
        help=("Score containing the solo or list of matchfiles"),
        nargs="+",
        default=solo_fn,
    )

    parser.add_argument(
        "--acc-fn",
        help=("Accompaniment score"),
        default=acc_fn,
    )

    parser.add_argument(
        "--init-bpm",
        type=float,
        default=init_bpm,
    )

    parser.add_argument(
        "--polling-period",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--midi-fn",
        default=midi_fn,
    )

    parser.add_argument(
        "--follower",
        default="OnlineTimeWarping",
    )

    parser.add_argument(
        "--live",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--accompaniment-match",
        default=accompaniment_match,
    )

    parser.add_argument(
        "--bypass_audio",
        default=False,
        help="bypass fluidsynth audio",
        action="store_true",
    )

    parser.add_argument(
        "--use_mediator",
        default=False,
        help="use ceus mediator",
        action="store_true",
    )

    args = parser.parse_args()

    """
    pass a port name or substring of a port name to the MIDI router to set up the ports
    use  mido.get_output_names() to get an idea of the names
    if a midi player is used the fourth port needs to loop back to the first

    pass a FluidSynthplayer  instance instead of a midi port name to the sound outputs to use fludisynth
    """

    # if audio is bypassed the mediator should be turned on as all
    # MIDI messages are sent to the same port maybe we can fix this by using
    # different ports for solo and accompaniment
    assert not args.bypass_audio or args.bypass_audio and args.use_mediator

    if PLATFORM in ("Darwin", "Linux") and not args.live:
        # Default for Carlos' Macbook ;)

        router_kwargs = dict(
            solo_input_to_accompaniment_port_name=0,
            acc_output_to_sound_port_name="IAC Driver Web Midi"
            if args.bypass_audio
            else FluidsynthPlayer,
            MIDIPlayer_to_sound_port_name="IAC Driver Web Midi"
            if args.bypass_audio
            else FluidsynthPlayer,
            MIDIPlayer_to_accompaniment_port_name=0,
            simple_button_input_port_name=None,
        )

    else:
        import mido

        args.midi_fn = None

        # TODO: check if the ports are the same in Linux and Windows
        live_instrument_ports = (
            "USB-MIDI",  # Essex in Oslo
            "Clavinova",  # Clavinova at WU
            "Silent Piano",  # Yamaha GB1 in Vienna
            "M-Audio MIDISPORT Uno",  # MIDI Interface
            "Scarlett 18i8 USB",  # Focusrite 18i8
            "Babyface (23665044) Port 1",
            "Babyface (23664861) Port 1",
            "Disklavier",
            "RD-88 1",
        )

        available_input_ports = mido.get_input_names()
        print("Available input ports")
        print(available_input_ports)
        port_intersection = set(available_input_ports).intersection(
            live_instrument_ports
        )
        if len(port_intersection) > 0:
            # Assume that there is only one live instrument
            instrument_port = list(port_intersection)[0]

        router_kwargs = dict(
            solo_input_to_accompaniment_port_name=instrument_port,
            acc_output_to_sound_port_name=instrument_port,
            MIDIPlayer_to_sound_port_name=None,
            MIDIPlayer_to_accompaniment_port_name=None,
            simple_button_input_port_name=None,
        )
    """
    example router for connection drom MIDI file player on port 0 and fluidsynth audio out

    fluidsynth = FluidsynthPlayer()

    midi_router = MidiRouter(
        solo_input_to_accompaniment_port_name = 0,
        acc_output_to_sound_port_name = fluidsynth,
        MIDIPlayer_to_sound_port_name = fluidsynth,
        MIDIPlayer_to_accompaniment_port_name = 0,
        simple_button_input_port_name = "your_controller_name",
    )

    example router windows

    midi_router = MidiRouter(
        solo_input_to_accompaniment_port_name = "acc_loopback",
        acc_output_to_sound_port_name = "cuit",
        MIDIPlayer_to_sound_port_name = "cuit",
        MIDIPlayer_to_accompaniment_port_name = "acc_loopback",
        simple_button_input_port_name = "MPK",
    )


    example router IAC with two ports "Bus" and "muse"
    midi_router = MidiRouter(
        solo_input_to_accompaniment_port_name = "Bus",
        acc_output_to_sound_port_name = "muse",
        MIDIPlayer_to_sound_port_name = "muse",
        MIDIPlayer_to_accompaniment_port_name = "Bus",
        simple_button_input_port_name = "your_controller_name",
    )
    """
    performance_codec_kwargs = (
        {
            "velocity_trend_ma_alpha": 0.6,
            "articulation_ma_alpha": 0.4,
            "velocity_dev_scale": 70,
            "velocity_min": 20,
            "velocity_max": 100,
            "velocity_solo_scale": 0.85,
            "timing_scale": 0.001,
            "log_articulation_scale": 0.1,
            "mechanical_delay": 0.0,
        },
    )

    # Setup scores
    solo_parts = []

    for i, fn in enumerate(solo_fn):

        if fn.endswith(".match"):
            if i == 0:
                solo_ppart, alignment, solo_spart = partitura.load_match(
                    fn=fn, create_part=True, first_note_at_zero=True
                )
            else:
                solo_ppart, alignment = partitura.load_match(
                    fn=fn, create_part=False, first_note_at_zero=True
                )

            ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
                ppart_or_note_array=solo_ppart,
                spart_or_note_array=solo_spart,
                alignment=alignment,
            )
            solo_parts.append((solo_ppart, ptime_to_stime_map, stime_to_ptime_map))
        else:
            solo_spart = partitura.load_musicxml(fn)

            if i == 0:
                solo_spart = solo_spart

            solo_parts.append((solo_spart, None, None))

    solo_score = part_to_score(solo_spart, bpm=init_bpm)

    if accompaniment_match is None:
        acc_spart = partitura.load_musicxml(acc_fn)
        acc_notes = list(part_to_score(acc_spart, bpm=init_bpm).notes)
        velocity_trend = None
        velocity_dev = None
        timing = None
        log_articulation = None
        log_bpr = None

    else:
        acc_ppart, acc_alignment, acc_spart = partitura.load_match(
            fn=accompaniment_match,
            first_note_at_zero=True,
            create_part=True,
        )
        acc_notes = list(
            alignment_to_score(
                fn_or_spart=acc_spart, ppart=acc_ppart, alignment=acc_alignment
            ).notes
        )
        pc = get_performance_codec(
            [
                "velocity_trend",
                "velocity_dev",
                "beat_period",
                "timing",
                "articulation_log",
            ]
        )
        bm_params, _, u_onset_idx = pc.encode(
            part=acc_spart,
            ppart=acc_ppart,
            alignment=acc_alignment,
            return_u_onset_idx=True,
        )

        bm_params_onsetwise = notewise_to_onsetwise(bm_params, u_onset_idx)

        # TODO Use the solo part to compute the moving average
        vt_ma = moving_average_offline(
            parameter=bm_params_onsetwise["velocity_trend"],
            alpha=performance_codec_kwargs.get("velocity_trend_ma_alpha", 0.6),
        )

        velocity_trend = onsetwise_to_notewise(
            bm_params_onsetwise["velocity_trend"] / vt_ma, u_onset_idx
        )

        if tempo_model.has_tempo_expectations:
            # get iterable of the tempo expectations
            tempo_model.tempo_expectations_func = interp1d(
                np.unique(acc_spart.note_array()["onset_beat"]),
                bm_params_onsetwise["beat_period"],
                bounds_error=False,
                kind="previous",
                fill_value=(
                    bm_params_onsetwise["beat_period"][0],
                    bm_params_onsetwise["beat_period"][-1],
                ),
            )
            init_bp = bm_params_onsetwise["beat_period"][0]

        vd_scale = performance_codec_kwargs.get("velocity_dev_scale", 90)
        velocity_dev = bm_params["velocity_dev"] * vd_scale

        timing_scale = performance_codec_kwargs.get("timing_scale", 1.0)
        timing = bm_params["timing"] * timing_scale
        print(np.mean(np.abs(timing)))
        lart_scale = performance_codec_kwargs.get("log_articulation_scale", 1.0)
        log_articulation = bm_params["articulation_log"] * lart_scale
        # log_articulation = None
        log_bpr = None

    acc_score = AccompanimentScore(
        notes=acc_notes,
        solo_score=solo_score,
        velocity_trend=velocity_trend,
        velocity_dev=velocity_dev,
        timing=timing,
        log_articulation=log_articulation,
        log_bpr=log_bpr,
    )

    pc = OnlinePerformanceCodec(
        beat_period_ave=init_bp,
        velocity_ave=64,
        init_eq_onset=0.0,
        tempo_model=tempo_model,
        **performance_codec_kwargs,
    )

    accompanion = ACCompanion()
    # accompanion = ACCompanion(
    #     solo_fn=args.solo_fn,
    #     acc_fn=args.acc_fn,
    #     router_kwargs=router_kwargs,
    #     tempo_model=tempo_model,
    #     midi_fn=args.midi_fn,
    #     init_bpm=args.init_bpm,
    #     polling_period=args.polling_period,
    #     follower=args.follower,
    #     follower_kwargs={"window_size": 100, "step_size": 10},
    #     ground_truth_match=None,
    #     accompaniment_match=args.accompaniment_match,
    #     use_ceus_mediator=args.use_mediator,
    #     # tempo_tapping=tempo_tapping,
    #     adjust_following_rate=0.2,
    #     # tap_tempo=False,
    #     bypass_audio=args.bypass_audio,
    # )

    try:
        accompanion.start()
    except KeyboardInterrupt:
        print("stop_playing")
        accompanion.stop_playing()
        accompanion.seq.panic_button()
    finally:
        accompanion.join()
