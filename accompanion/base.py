# -*- coding: utf-8 -*-
"""
ACCompanion!
"""
import multiprocessing
import threading
import time
import numpy as np
import os
from accompanion.config import CONFIG
from typing import Optional

from accompanion.midi_handler.midi_input import create_midi_poll, POLLING_PERIOD
from accompanion.midi_handler.midi_file_player import get_midi_file_player
from accompanion.midi_handler.midi_sequencing_threads import ScoreSequencer
from accompanion.midi_handler.midi_routing import (
    MidiRouter,
    DummyRouter,
    RecordingRouter,
)
from accompanion.midi_handler.midi_utils import midi_file_from_midi_msg

from accompanion.mtchmkr.utils_generic import SequentialOutputProcessor

from accompanion.accompanist.score import AccompanimentScore, Score

from accompanion.accompanist.accompaniment_decoder import (
    OnlinePerformanceCodec,
    Accompanist,
)

from accompanion.accompanist.tempo_models import SyncModel
from accompanion.midi_handler.ceus_mediator import CeusMediator
from accompanion.score_follower.note_tracker import NoteTracker
from accompanion.score_follower.onset_tracker import OnsetTracker, DiscreteOnsetTracker
from accompanion.score_follower.trackers import AccompanimentScoreFollower
from accompanion.midi_handler.fluid import FluidsynthPlayer


ACC_PARENT = multiprocessing.Process if CONFIG["ACC_PROCESS"] else threading.Thread


class ACCompanion(ACC_PARENT):
    """
    Main class for running the ACCompanion.

    Both the HMMACCompanion and the OLTWACCompanion inherit from this class.
    Arguments of this class are initialized on methods of the child classes.

    Parameters
    ----------
    solo_score: Score
        Score object for the solo part.
    accompaniment_score: AccompanimentScore
        Score object for the accompaniment part.
    score_follower: AccompanimentScoreFollower
        Score follower object for the accompaniment part.
    tempo_model: SyncModel
        Tempo model object for the accompaniment part.
    performance_codec: OnlinePerformanceCodec
        Performance codec object for the accompaniment part.
    input_pipeline: SequentialOutputProcessor
        Input pipeline object for the accompaniment part.
    midi_router: MidiRouter
        Midi router object for handling MIDI messages.
    midi_fn: str (optional)
        Path to the MIDI file to be played.
    init_bpm: float = 60
        Initial tempo in beats per minute.
    init_velocity: int = 60
        Initial velocity for the MIDI messages.
    polling_period: float
        Period of the polling loop in seconds.
    use_ceus_mediator: bool
        Whether to use the CEUS mediator.
    adjust_following_rate: float
        A float between 0 and 1. The rate at which the score follower ...
    bypass_audio: bool = False
        Bypass fluidsynth audio
    test: bool = False
        switch to Dummy MIDI ROuter for test environment
    record_midi_path : str
    """

    def __init__(
        self,
        score_kwargs: dict,
        score_follower_kwargs: dict,
        tempo_model_kwargs: dict,
        performance_codec_kwargs: dict,  # this is just a workaround for now
        midi_router_kwargs: dict,  # this is just a workaround for now
        midi_fn: Optional[str] = None,
        init_bpm: float = 60,
        init_velocity: int = 60,
        polling_period: float = POLLING_PERIOD,
        use_ceus_mediator: bool = False,
        adjust_following_rate: float = 0.1,
        onset_tracker_type: str = "continuous",
        bypass_audio: bool = False,  # bypass fluidsynth audio
        test: bool = False,  # switch to Dummy MIDI ROuter for test environment
        record_midi: bool = False,
    ) -> None:
        super(ACCompanion, self).__init__()

        self.performance_codec_kwargs = performance_codec_kwargs
        self.score_kwargs = score_kwargs
        self.score_follower_kwargs = score_follower_kwargs
        self.tempo_model_kwargs = tempo_model_kwargs
        self.solo_score: Optional[Score] = None
        self.acc_score: Optional[AccompanimentScore] = None
        self.accompanist = None

        self.midi_fn: Optional[str] = midi_fn

        self.router_kwargs = midi_router_kwargs

        self.use_mediator: bool = use_ceus_mediator
        self.mediator: Optional[CeusMediator] = None

        self.init_bpm: float = init_bpm
        self.init_bp: float = 60 / self.init_bpm
        self.init_velocity: int = init_velocity
        self.beat_period = self.init_bp
        self.velocity = self.init_velocity
        self.record_midi = record_midi

        # Parameters for following
        self.polling_period: float = polling_period
        self.score_follower: Optional[AccompanimentScoreFollower] = None
        self.tempo_model = None
        self.bypass_audio: bool = True if test else bypass_audio
        self.play_accompanion: bool = False
        # Rate in "loops_without_update"  for adjusting the score
        self.adjust_following_rate: float = adjust_following_rate
        # follower with expected position at the current tempo.
        self.afr: float = np.round(1 / self.polling_period * self.adjust_following_rate)
        self.input_pipeline = None
        self.seq = None
        self.note_tracker = None
        self.pipe_out = None
        self.queue = None
        self.midi_input_process = None
        self.router = None
        self.dummy_solo = None
        self.test = test
        self.onset_tracker_type = onset_tracker_type

    def setup_scores(self) -> None:
        """Method to be overwritten by the child classes."""
        raise NotImplementedError

    def setup_accompanist(self) -> None:
        """Method to be overwritten by the child classes."""
        raise NotImplementedError

    def setup_score_follower(self) -> None:
        """Method to be overwritten by the child classes."""
        raise NotImplementedError

    def check_empty_frames(self, frame) -> bool:
        """Method to be overwritten by the child classes."""
        raise NotImplementedError

    def setup_process(self):
        """
        Setup the process for the ACCompanion.
        """
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

        self.setup_scores()
        self.setup_score_follower()
        self.performance_codec = OnlinePerformanceCodec(
            beat_period_ave=self.init_bp,
            velocity_ave=self.velocity,
            init_eq_onset=0.0,
            tempo_model=self.tempo_model,
            **self.performance_codec_kwargs,
        )

        self.accompanist: Accompanist = Accompanist(
            accompaniment_score=self.acc_score,
            performance_codec=self.performance_codec,
        )

        if self.use_mediator:
            self.mediator = CeusMediator()

        if self.test:
            self.router = DummyRouter(**self.router_kwargs)
        elif self.record_midi:

            if isinstance(self.score_kwargs["solo_fn"], (list, tuple)):
                piece_name = self.score_kwargs["solo_fn"][0].split(os.path.sep)[-2]
            elif isinstance(self.score_kwargs["solo_fn"], str):
                piece_name = self.score_kwargs["solo_fn"].split(os.path.sep)[-2]
            else:
                raise ValueError(
                    f"{self.score_kwargs['solo_fn']} should be a string or a list"
                )
            self.router = RecordingRouter(self.router_kwargs, piece_name)
        else:
            self.router = MidiRouter(**self.router_kwargs)

        self.seq: ScoreSequencer = ScoreSequencer(
            score_or_notes=self.acc_score,
            outport=self.router.acc_output_to_sound_port,
            mediator=self.mediator,
        )

        self.seq.panic_button()

        # Update tempo model
        self.tempo_model.beat_period = self.init_bp
        self.prev_score_onset: float = self.solo_score.unique_onsets.min()
        self.first_score_onset: float = self.solo_score.unique_onsets.min()

        # initialize note tracker
        self.note_tracker: NoteTracker = NoteTracker(self.solo_score.note_array)
        self.accompanist.pc.note_tracker = self.note_tracker

        self.pipe_out, self.queue, self.midi_input_process = create_midi_poll(
            port=self.router.solo_input_to_accompaniment_port,
            polling_period=self.polling_period,
            # velocities only for visualization purposes
            pipeline=self.input_pipeline,
            return_midi_messages=True,
            thread=CONFIG["USE_THREADS"],
            mediator=self.mediator,
        )

    @property
    def beat_period(self) -> float:
        return self.beat_period_

    @beat_period.setter
    def beat_period(self, beat_period: float) -> None:
        """
        Sets a new value for the beat period and updates the accompanist
        """
        self.beat_period_: float = beat_period

        if self.accompanist is not None:
            self.accompanist.pc.bp_ave = beat_period

    @property
    def velocity(self) -> int:
        return self.velocity_

    @velocity.setter
    def velocity(self, velocity: int) -> None:
        """
        Set a new value for the MIDI velocity

        Parameters
        ----------
        velocity: int
            MIDI velocity
        """
        self.velocity_ = velocity

    def stop_playing(self) -> None:
        """
        Stops ACCompanion
        """
        self.play_accompanion = False
        if self.dummy_solo is not None:
            self.dummy_solo.stop_playing()
            self.dummy_solo.join()
        self.midi_input_process.stop_listening()
        self.seq.stop_playing()
        self.seq.panic_button()
        self.router.close_ports()
        self.seq.join()
        self.midi_input_process.join()
        print("All processes have finished")

    def terminate(self):
        """
        Terminate process of the ACCompanion
        """
        self.stop_playing()

    def run(self):
        """
        Main run method
        """
        self.setup_process()
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

        if self.onset_tracker_type == "discrete":
            onset_tracker = DiscreteOnsetTracker(self.solo_score.unique_onsets)
        else:
            onset_tracker = OnsetTracker(self.solo_score.unique_onsets)

        # TODO: Initialize on-line Basis Mixer here
        # expression_model = BasisMixer()
        self.midi_input_process.start()
        print("Start listening")

        self.perf_frame = None

        if self.midi_fn is not None:
            print("Start playing MIDI file")
            self.dummy_solo = get_midi_file_player(
                port=self.router.MIDIPlayer_to_accompaniment_port,
                file_name=self.midi_fn,
                player_class=FluidsynthPlayer,
                thread=CONFIG["USE_THREADS"],
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

        test_counter = 0

        try:
            while not self.seq.end_of_piece:
                # TODO GH Issue 22: "if not self.queue.poll()"
                # vs. "self.queue.poll() is not None"
                if self.queue.poll() is not None:
                    output = self.queue.recv()
                    solo_p_onset = time.time() - start_time
                    input_midi_messages, output = output
                    # Use these onset times?
                    onset_times = [
                        msg[1]
                        for msg in input_midi_messages
                        if msg[0].type in ("note_on", "note_off")
                    ]
                    onset_time = np.mean(onset_times) if len(onset_times) > 0 else 0
                    new_midi_messages = False
                    decay *= CONFIG["DECAY_VALUE"]
                    for msg, msg_time in input_midi_messages:
                        if msg.type in ("note_on", "note_off"):

                            if msg.type == "note_on" and msg.velocity > 0:
                                new_midi_messages = True
                            midi_msg = (msg.type, msg.note, msg.velocity, onset_time)
                            self.note_tracker.track_note(midi_msg)

                            decay[msg.note - 21] = 1.0

                    if self.check_empty_frames(output):
                        empty_loops += 1
                    else:
                        empty_loops = 0

                    # if perf_start:
                    score_position = self.score_follower(output)

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

                        if self.test and test_counter == 10:
                            break
                        else:
                            test_counter += 1

                        if not acc_update:
                            asynch = expected_position - solo_s_onset
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
        except Exception as e:
            print(e)
            pass
        finally:
            self.stop_playing()
