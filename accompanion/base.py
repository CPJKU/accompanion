# -*- coding: utf-8 -*-
"""
ACCompanion!
"""
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from accompanion.accompanist.accompaniment_decoder import (
    Accompanist,
    OnlinePerformanceCodec,
)
from accompanion.accompanist.fermata import (
    FermataHold,
    merge_sections,
    waiting_points,
)
from accompanion.accompanist.score import AccompanimentScore, Score
from accompanion.config import CONFIG
from accompanion.midi_handler.ceus_mediator import CeusMediator
from accompanion.midi_handler.fluid import FluidsynthPlayer
from accompanion.midi_handler.midi_file_player import get_midi_file_player
from accompanion.midi_handler.midi_input import POLLING_PERIOD, create_midi_poll
from accompanion.midi_handler.midi_routing import (
    DummyRouter,
    MidiRouter,
    RecordingRouter,
)
from accompanion.midi_handler.midi_sequencing_threads import ScoreSequencer
from accompanion.score_follower.note_tracker import NoteTracker
from accompanion.score_follower.onset_tracker import DiscreteOnsetTracker, OnsetTracker
from accompanion.score_follower.trackers import (
    AccompanimentScoreFollower,
    ExpectedPositionTracker,
)

ACC_PARENT = multiprocessing.Process if CONFIG["ACC_PROCESS"] else threading.Thread


@dataclass
class FollowingState:
    """What the following loop carries from one input frame to the next.

    Held in an object rather than in local variables so that `follow_step` can
    be driven from outside `run` -- by the offline harness in `bin/`, which
    measures the accompaniment without MIDI hardware and must exercise exactly
    the code that plays live.
    """

    #: Dead-reckoned score position, advanced by the elapsed time over the
    #: current beat period and pulled back toward the follower on every onset.
    expected_position: float
    prev_solo_p_onset: Optional[float] = None
    loops_without_update: int = 0
    empty_loops: int = 0
    adjusted_sf: bool = False
    acc_step_counter: int = 0
    #: Whether the solo part starts before the accompaniment.
    solo_starts: bool = True
    sequencer_start: bool = False


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
    input_pipeline: matchmaker.features.processor.Processor
        Feature processor applied to the incoming MIDI frames.
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
    fermata_kwargs: dict (optional)
        How the accompaniment waits at fermatas and in free sections. See
        `setup_fermata_hold` for the keys, and
        `accompanion.accompanist.fermata` for what waiting means. Waiting is
        on by default; ``{"enabled": False}`` counts through fermatas the way
        the ACCompanion did before.
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
        expected_position_weight: float = 0.6,
        onset_tracker_type: str = "continuous",
        bypass_audio: bool = False,  # bypass fluidsynth audio
        test: bool = False,  # switch to Dummy MIDI ROuter for test environment
        record_midi: bool = False,
        accompanist_decoder_kwargs: Optional[dict] = None,
        fermata_kwargs: Optional[dict] = None,
    ) -> None:
        super(ACCompanion, self).__init__()

        self.performance_codec_kwargs = performance_codec_kwargs
        self.score_kwargs = score_kwargs
        self.score_follower_kwargs = score_follower_kwargs
        self.tempo_model_kwargs = tempo_model_kwargs
        self.solo_score: Optional[Score] = None
        self.acc_score: Optional[AccompanimentScore] = None
        self.accompanist_decoder_kwargs: Optional[dict] = accompanist_decoder_kwargs
        self.accompanist = None
        self.time_delays = list()
        self.alignment = list()
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

        # Expected position tracker
        self.expected_position_tracker: Optional[ExpectedPositionTracker] = None
        # Rate in "loops_without_update"  for adjusting the score
        self.adjust_following_rate: float = adjust_following_rate
        self.expected_position_weight: float = expected_position_weight
        # follower with expected position at the current tempo.
        self.afr: float = np.round(1 / self.polling_period * self.adjust_following_rate)
        self.input_pipeline = None
        # Set by `setup_score_follower` for followers that align note by note
        # and cannot take a frame holding a chord.
        self.event_based_input: bool = False
        self.seq = None
        self.note_tracker = None
        self.pipe_out = None
        self.queue = None
        self.midi_input_process = None
        self.router = None
        self.dummy_solo = None
        self.test = test
        self.onset_tracker_type = onset_tracker_type
        self.fermata_kwargs: dict = fermata_kwargs or {}
        # Built by `setup_following`, once the scores are known.
        self.fermata_hold: Optional[FermataHold] = None

        print("expected_position_weight", self.expected_position_weight)

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

    def setup_following(self) -> None:
        """Build everything the following loop needs, and nothing else.

        The scores, the score follower, the accompanist and the trackers --
        but no MIDI ports, no sequencer and no audio. Split out of
        `setup_process` so that the offline harness in `bin/` can measure the
        accompaniment without any hardware.
        """
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
            decoder_kwargs=self.accompanist_decoder_kwargs,
        )

        # Update tempo model
        self.tempo_model.beat_period = self.init_bp
        self.prev_score_onset: float = self.solo_score.unique_onsets.min()
        self.first_score_onset: float = self.solo_score.unique_onsets.min()

        # initialize note tracker
        self.note_tracker: NoteTracker = NoteTracker(self.solo_score.note_array)
        self.accompanist.pc.note_tracker = self.note_tracker

        self.expected_position_tracker = ExpectedPositionTracker(
            tempo_model=self.tempo_model,
            first_onset=self.first_score_onset,
        )

        self.fermata_hold = self.setup_fermata_hold()

    def setup_fermata_hold(self) -> FermataHold:
        """Work out where the accompaniment waits for the soloist.

        The score is the first source: partitura reads both the fermatas and
        the words that mark a passage as free, and `part_to_score` puts them
        on the solo `Score`. `fermata_kwargs` adds to what the score says, for
        an engraving that leaves a fermata out, or a free passage marked only
        in the part the ACCompanion never sees.

        Recognised keys, all optional:

        ``enabled``
            Whether to wait at all. Default True.
        ``fermata_onsets``
            Extra waiting points, in score beats.
        ``free_sections``
            Extra free spans, as ``[start, end]`` pairs in score beats. Every
            solo onset in the span becomes a waiting point, so the passage is
            taken one note at a time.
        ``detect_free_sections``
            Whether to read free sections off the words in the score, which is
            a guess at what a marking such as *cadenza* covers. Default True.
            The fermatas themselves are never guessed.
        ``max_silence``, ``max_lost``, ``sustain_margin``, ``verbose``
            Passed to `FermataHold`.
        """
        kwargs = dict(self.fermata_kwargs)
        enabled = kwargs.pop("enabled", True)
        extra_fermatas = kwargs.pop("fermata_onsets", None) or []
        extra_sections = kwargs.pop("free_sections", None) or []
        detect_free_sections = kwargs.pop("detect_free_sections", True)

        fermata_onsets = np.r_[
            np.asarray(self.solo_score.fermata_onsets, dtype=float),
            np.asarray(extra_fermatas, dtype=float),
        ]
        free_sections = merge_sections(
            (list(self.solo_score.free_sections) if detect_free_sections else [])
            + [tuple(section) for section in extra_sections]
        )

        onsets = (
            waiting_points(
                solo_onsets=self.solo_score.unique_onsets,
                fermata_onsets=fermata_onsets,
                free_sections=free_sections,
            )
            if enabled
            else []
        )

        hold = FermataHold(
            acc_notes=self.acc_score.notes,
            onsets=onsets,
            solo_onsets=self.solo_score.unique_onsets,
            **kwargs,
        )

        if len(hold):
            listed = ", ".join(f"{onset:g}" for onset in hold.onsets[:12])
            if len(hold) > 12:
                listed += ", ..."
            print(f"Waiting for the soloist at {len(hold)} onsets: {listed}")
            for start, end in free_sections:
                print(f"  free section: beats {start:g} to {end:g}")
        elif enabled and (len(fermata_onsets) or free_sections):
            print(
                "This score's fermatas are all in places nothing can be "
                "waited for: over a rest, or on the final chord"
            )

        return hold

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

        self.setup_following()

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
            self.router = RecordingRouter(piece_name, **self.router_kwargs)
        else:
            self.router = MidiRouter(**self.router_kwargs)

        self.seq: ScoreSequencer = ScoreSequencer(
            score_or_notes=self.acc_score,
            outport=self.router.acc_output_to_sound_port,
            mediator=self.mediator,
        )

        self.seq.panic_button()

        self.pipe_out, self.queue, self.midi_input_process = create_midi_poll(
            port=self.router.solo_input_to_accompaniment_port,
            polling_period=None if self.event_based_input else self.polling_period,
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

    def follow_step(
        self,
        input_midi_messages,
        output,
        solo_p_onset: float,
        onset_tracker,
        state: "FollowingState",
    ) -> bool:
        """Advance the following loop by one input frame.

        Tracks the incoming notes, turns the score follower's position into a
        score onset, and steps the accompanist. Split out of `run` so that the
        offline harness (`bin/test_accompaniment.py`) drives exactly the code
        that plays live.

        At a fermata, or anywhere inside a free section, score time stops
        here: the frame is spent keeping the held chord sounding, and neither
        the dead reckoning nor the accompaniment moves until the soloist
        plays the next onset. See `accompanion.accompanist.fermata`.

        Parameters
        ----------
        input_midi_messages : list
            The `(message, time)` pairs of this frame.
        output : Any
            The input pipeline's output for this frame.
        solo_p_onset : float
            Time of this frame, in seconds since the performance started.
        onset_tracker : OnsetTracker or DiscreteOnsetTracker
        state : FollowingState
            Carried across frames, and updated in place.

        Returns
        -------
        start_sequencer : bool
            True on the frame where the accompaniment should start playing.
        """
        start_sequencer = False
        new_midi_messages = False

        for msg, msg_time in input_midi_messages:
            if msg.type in ("note_on", "note_off"):

                if msg.type == "note_on" and msg.velocity > 0:
                    new_midi_messages = True
                midi_msg = (msg.type, msg.note, msg.velocity, solo_p_onset)
                self.note_tracker.track_note(midi_msg)

        if self.check_empty_frames(output):
            state.empty_loops += 1
        else:
            state.empty_loops = 0

        score_position = self.score_follower(output)
        solo_s_onset, onset_index, acc_update = onset_tracker(
            score_position,
            state.expected_position,
        )

        pioi = (
            solo_p_onset - state.prev_solo_p_onset
            if state.prev_solo_p_onset is not None
            else self.polling_period
        )
        state.prev_solo_p_onset = solo_p_onset

        waiting = self.fermata_hold.waiting
        if waiting and solo_s_onset is None:
            # Still at the fermata. The soloist counts as present while a key
            # is down or a note has just arrived; a wait that outlasts their
            # silence is a breakdown rather than a held note.
            if self.fermata_hold.keep_waiting(
                perf_onset=solo_p_onset,
                holding=bool(self.note_tracker.open_notes),
                new_notes=new_midi_messages,
            ):
                # Hold the follower on the fermata too: the gap the soloist
                # is opening up is not an inter-onset interval the score can
                # explain, and a follower that reads it as one walks away from
                # a soloist who has not moved at all.
                self.score_follower.discount_wait(
                    perf_time=solo_p_onset,
                    expected_ioi=self.fermata_hold.score_gap * self.beat_period,
                )
                return start_sequencer
            self.fermata_hold.give_up(solo_p_onset, self.beat_period)
            waiting = False

        if waiting:
            # The soloist has moved on. Score time resumes here: the wait took
            # no score time at all, so the dead reckoning is set down on the
            # onset they played rather than being nudged towards it, and the
            # tempo model is re-anchored instead of reading the wait as tempo.
            self.fermata_hold.resume(solo_p_onset, solo_s_onset, self.beat_period)
            self.tempo_model.resync(solo_p_onset, solo_s_onset)
            state.expected_position = solo_s_onset
        else:
            state.expected_position = state.expected_position + pioi / self.beat_period

        if solo_s_onset is not None:

            print(
                f"performed onset {solo_s_onset}",
                f"expected onset {self.expected_position_tracker.expected_position}",
                f"beat_period {self.beat_period}",
                f"adjusted {acc_update or state.adjusted_sf}",
            )

            self.time_delays.append([solo_s_onset, solo_p_onset, self.beat_period])

            if not acc_update:
                self.expected_position_tracker.expected_position = solo_s_onset
                asynch = state.expected_position - solo_s_onset
                state.expected_position = (
                    state.expected_position - self.expected_position_weight * asynch
                )
                state.loops_without_update = 0
                state.adjusted_sf = False
            else:
                state.loops_without_update += 1

            if new_midi_messages:
                self.note_tracker.update_alignment(solo_s_onset)
            # start accompaniment if it starts at the
            # same time as the solo
            if state.solo_starts and onset_index == 0:
                if not state.sequencer_start:
                    state.sequencer_start = True
                    self.accompanist.accompaniment_step(
                        solo_s_onset=solo_s_onset,
                        solo_p_onset=solo_p_onset,
                    )
                    start_sequencer = True

            if (
                solo_s_onset > self.first_score_onset
                and not acc_update
                and not state.adjusted_sf
            ):
                print(f"step {state.acc_step_counter} {solo_s_onset}")
                self.accompanist.accompaniment_step(
                    solo_s_onset=solo_s_onset, solo_p_onset=solo_p_onset
                )
                self.beat_period = self.accompanist.pc.bp_ave
                state.acc_step_counter += 1

            # A fermata, or a note inside a free section. The accompanist has
            # just scheduled the whole rest of the piece from here at the
            # tempo the soloist arrived with; everything past this onset is
            # suspended again until they move on.
            self.fermata_hold.begin(solo_s_onset, solo_p_onset)
        else:
            state.loops_without_update += 1

        if not self.fermata_hold.waiting and state.loops_without_update % self.afr == 0:
            # only allow forward updates
            if self.score_follower.current_position < state.expected_position:
                self.score_follower.update_position(state.expected_position)
                state.adjusted_sf = True

        return start_sequencer

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

        state = FollowingState(
            expected_position=self.first_score_onset,
            solo_starts=solo_starts,
            sequencer_start=sequencer_start,
        )

        try:
            while not self.seq.end_of_piece:
                # TODO GH Issue 22: "if not self.queue.poll()"
                # vs. "self.queue.poll() is not None"
                # (NV) actually, this should be "have a CORRECT branch (non-blocking MIDI)
                # vs. no branch (blocking MIDI)"

                # this version of recv uses the quasi-blocking version with
                # periodic timeouts
                output = self.queue.recv()
                solo_p_onset = time.time() - start_time
                input_midi_messages, output = output

                if self.follow_step(
                    input_midi_messages=input_midi_messages,
                    output=output,
                    solo_p_onset=solo_p_onset,
                    onset_tracker=onset_tracker,
                    state=state,
                ):
                    print("Start accompaniment")
                    self.seq.start()
            self.alignment = self.note_tracker.alignment
        except Exception as e:
            print(e)
            pass
        finally:
            self.stop_playing()
