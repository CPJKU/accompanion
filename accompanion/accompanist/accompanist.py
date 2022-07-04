# -*- coding: utf-8 -*-
"""
ACCompanion!
"""
import multiprocessing
import threading
import time

import numpy as np
import partitura

from basismixer.performance_codec import get_performance_codec
from basismixer.utils.music import onsetwise_to_notewise, notewise_to_onsetwise
from scipy.interpolate import interp1d

from accompanion.midi_handler.midi_input import create_midi_poll, POLLING_PERIOD
from accompanion.midi_handler.midi_file_player import get_midi_file_player
from accompanion.midi_handler.midi_sequencing_threads import ScoreSequencer
from accompanion.midi_handler.midi_routing import MidiRouter

from accompanion.mtchmkr.features_midi import PianoRollProcessor
from accompanion.mtchmkr.alignment_online_oltw import (
    OnlineTimeWarping,
)

from accompanion.mtchmkr.utils_generic import SequentialOutputProcessor

from accompanion.accompanist.score import (
    part_to_score,
    alignment_to_score,
    AccompanimentScore,
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
from accompanion.score_follower.trackers import MultiDTWScoreFollower
from accompanion.midi_handler.fluid import FluidsynthPlayer


ACC_PROCESS = True
ACC_PARENT = multiprocessing.Process if ACC_PROCESS else threading.Thread
USE_THREADS = True


class ACCompanion(ACC_PARENT):
    def __init__(
        self,
        solo_fn,
        acc_fn,
        midi_fn=None,
        init_bpm=60,
        init_velocity=60,
        polling_period=POLLING_PERIOD,
        follower="OnlineTimeWarping",
        follower_kwargs={"window_size": 80, "step_size": 10},
        ground_truth_match=None,
        router_kwargs={},
        tempo_model=tempo_models.LSM,
        tempo_model_kwargs={},
        accompaniment_match=None,
        # tap_tempo=False,
        pipe=None,
        use_ceus_mediator=False,
        performance_codec_kwargs={
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
        adjust_following_rate=0.1,
        # tempo_tapping=None,
        bypass_audio=False,  # bypass fluidsynth audio
    ):
        super(ACCompanion, self).__init__()

        self.solo_fn = solo_fn
        self.acc_fn = acc_fn
        self.midi_fn = midi_fn

        # Matchfile with ground truth alignment (to
        # test the accuracy of the score follower)
        self.ground_truth_match = ground_truth_match
        # Matchfile with precomputed performance from the
        # Basis Mixer
        self.accompaniment_match = accompaniment_match

        self.router = None
        self.router_kwargs = router_kwargs

        self.init_bpm = init_bpm
        self.init_bp = 60 / self.init_bpm
        self.init_velocity = init_velocity

        # Parameters for following
        self.polling_period = polling_period

        self.follower_type = follower
        self.follower_kwargs = follower_kwargs

        self.tempo_model_kwargs = tempo_model_kwargs
        self.tempo_model = tempo_model
        self.performance_codec_kwargs = performance_codec_kwargs

        self.solo_parts = []
        self.solo_spart, self.solo_score, self.acc_score = None, None, None

        # MIDI communication
        self.pipe_out, self.queue, self.midi_input_process = None, None, None

        # reference features and frame index in reference for visualization
        self.reference_features = None
        self.score_idx = 0
        self.perf_frame = None

        self.score_follower = None
        self.note_tracker = None

        self.accompanist = None
        self.seq = None
        self.mediator = None
        self.use_mediator = use_ceus_mediator

        self.dummy_solo = None

        self.reference_is_performance = False
        self.play_accompanion = False

        self.beat_period = self.init_bp

        self.prev_score_onset = None

        # self.tap_tempo = tap_tempo
        # self.tempo_tapping = tempo_tapping
        self.tempo_counter = 0
        self.prev_metro_time = None
        self.tempo_sum = 0
        self.pipe = pipe
        self.first_score_onset = None
        self.adjust_following_rate = adjust_following_rate
        # Rate in "loops_without_update"  for adjusting the score
        # follower with expected position at the
        # current tempo
        self.afr = np.round(1 / self.polling_period * self.adjust_following_rate)

        # bypass fluidsynth audio
        self.bypass_audio = bypass_audio

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

    def setup_scores(self):
        """
        Load scores and prepare the accompaniment
        """

        self.solo_parts = []

        for i, fn in enumerate(self.solo_fn):

            if fn.endswith(".match"):
                if i == 0:
                    solo_ppart, alignment, self.solo_spart = partitura.load_match(
                        fn=fn, create_part=True, first_note_at_zero=True
                    )
                else:
                    solo_ppart, alignment = partitura.load_match(
                        fn=fn, create_part=False, first_note_at_zero=True
                    )

                ptime_to_stime_map, stime_to_ptime_map = get_time_maps_from_alignment(
                    ppart_or_note_array=solo_ppart,
                    spart_or_note_array=self.solo_spart,
                    alignment=alignment,
                )
                self.solo_parts.append(
                    (solo_ppart, ptime_to_stime_map, stime_to_ptime_map)
                )
            else:
                solo_spart = partitura.load_musicxml(fn)

                if i == 0:
                    self.solo_spart = solo_spart

                self.solo_parts.append((solo_spart, None, None))

        self.solo_score = part_to_score(self.solo_spart, bpm=self.init_bpm)

        if self.accompaniment_match is None:
            acc_spart = partitura.load_musicxml(self.acc_fn)
            acc_notes = list(part_to_score(acc_spart, bpm=self.init_bpm).notes)
            velocity_trend = None
            velocity_dev = None
            timing = None
            log_articulation = None
            log_bpr = None

        else:
            acc_ppart, acc_alignment, acc_spart = partitura.load_match(
                fn=self.accompaniment_match,
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
                alpha=self.performance_codec_kwargs.get("velocity_trend_ma_alpha", 0.6),
            )

            velocity_trend = onsetwise_to_notewise(
                bm_params_onsetwise["velocity_trend"] / vt_ma, u_onset_idx
            )

            if self.tempo_model.has_tempo_expectations:
                # get iterable of the tempo expectations
                self.tempo_model.tempo_expectations_func = interp1d(
                    np.unique(acc_spart.note_array()["onset_beat"]),
                    bm_params_onsetwise["beat_period"],
                    bounds_error=False,
                    kind="previous",
                    fill_value=(
                        bm_params_onsetwise["beat_period"][0],
                        bm_params_onsetwise["beat_period"][-1],
                    ),
                )
                self.init_bp = bm_params_onsetwise["beat_period"][0]

            vd_scale = self.performance_codec_kwargs.get("velocity_dev_scale", 90)
            velocity_dev = bm_params["velocity_dev"] * vd_scale

            timing_scale = self.performance_codec_kwargs.get("timing_scale", 1.0)
            timing = bm_params["timing"] * timing_scale
            print(np.mean(np.abs(timing)))
            lart_scale = self.performance_codec_kwargs.get(
                "log_articulation_scale", 1.0
            )
            log_articulation = bm_params["articulation_log"] * lart_scale
            # log_articulation = None
            log_bpr = None

        self.acc_score = AccompanimentScore(
            notes=acc_notes,
            solo_score=self.solo_score,
            velocity_trend=velocity_trend,
            velocity_dev=velocity_dev,
            timing=timing,
            log_articulation=log_articulation,
            log_bpr=log_bpr,
        )

        pc = OnlinePerformanceCodec(
            beat_period_ave=self.init_bp,
            velocity_ave=64,
            init_eq_onset=0.0,
            tempo_model=self.tempo_model,
            **self.performance_codec_kwargs,
        )

        self.accompanist = Accompanist(
            accompaniment_score=self.acc_score, performance_codec=pc
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

        # initialize note tracker
        self.note_tracker = NoteTracker(self.solo_spart.note_array())
        self.accompanist.pc.note_tracker = self.note_tracker

    def setup_score_follower(self):
        pipeline = SequentialOutputProcessor([PianoRollProcessor(piano_range=True)])

        state_to_ref_time_maps = []
        ref_to_state_time_maps = []
        score_followers = []

        # reference score for visualization
        self.reference_features = None

        for part, state_to_ref_time_map, ref_to_state_time_map in self.solo_parts:

            if state_to_ref_time_map is not None:
                ref_frames = partitura_to_framed_midi(
                    part_or_notearray_or_filename=part,
                    is_performance=True,
                    pipeline=pipeline,
                    polling_period=self.polling_period,
                )[0]

            else:
                raise NotImplementedError

            state_to_ref_time_maps.append(state_to_ref_time_map)
            ref_to_state_time_maps.append(ref_to_state_time_map)
            ref_features = np.array(ref_frames).astype(float)

            if self.reference_features is None:
                self.reference_features = ref_features

            # setup score follower
            if self.follower_type == "OnlineTimeWarping":
                score_follower = OnlineTimeWarping(
                    reference_features=ref_features, **self.follower_kwargs
                )
            else:
                raise NotImplementedError

            score_followers.append(score_follower)

        self.score_follower = MultiDTWScoreFollower(
            score_followers,
            state_to_ref_time_maps,
            ref_to_state_time_maps,
            self.polling_period,
        )

        self.pipe_out, self.queue, self.midi_input_process = create_midi_poll(
            port_name=self.router.solo_input_to_accompaniment_port_name[1],
            polling_period=self.polling_period,
            # velocities only for visualization purposes
            pipeline=SequentialOutputProcessor(
                [PianoRollProcessor(piano_range=True, use_velocity=True)]
            ),
            return_midi_messages=True,
            thread=USE_THREADS,
            mediator=self.mediator,
        )

    def get_reference_features(self):
        return self.reference_features

    def get_performance_frame(self):
        return self.perf_frame

    def get_accompaniment_frame(self):
        return self.seq.get_midi_frame()

    def get_score_index(self):
        return self.score_idx

    def stop_playing(self):
        self.play_accompanion = False

        if self.dummy_solo is not None:
            self.dummy_solo.stop_playing()

        self.midi_input_process.stop_listening()
        self.seq.stop_playing()
        self.router.close_ports()
        # self.join()

    def terminate(self):

        if hasattr(super(ACCompanion, self), "terminate"):
            super(ACCompanion, self).terminate()
        else:
            self.stop_playing()

    def get_tempo(self):
        return 60 / self.beat_period

    def run(self):

        if self.router_kwargs.get("acc_output_to_sound_port_name", None) is not None:
            try:
                # For SynthPorts
                self.router_kwargs[
                    "acc_output_to_sound_port_name"
                ] = self.router_kwargs["acc_output_to_sound_port_name"]()
            except TypeError:
                pass

        if self.router_kwargs.get("MIDIPlayer_to_sound_port_name", None) is not None:
            try:
                self.router_kwargs[
                    "MIDIPlayer_to_sound_port_name"
                ] = self.router_kwargs["MIDIPlayer_to_sound_port_name"]()
            except TypeError:
                pass

        self.router = MidiRouter(**self.router_kwargs)
        self.tempo_model = self.tempo_model(**self.tempo_model_kwargs)
        self.setup_scores()
        self.setup_score_follower()
        self.tempo_model.prev_score_onset = self.solo_score.min_onset

        if self.pipe is not None:
            self.pipe.send(self.get_reference_features())

        self.play_accompanion = True
        solo_starts = True
        sequencer_start = False
        start_time = None
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
                    output[output > 0] = 1.0
                    
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

                        if not acc_update:
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
