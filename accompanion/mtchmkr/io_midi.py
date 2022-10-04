# -*- coding: utf-8 -*-
"""
This module provides basic functionality to process MIDI inputs in
real time. 

# Changes to old implementation
- platform specifics: streamlined process of finding the current system's OS 

# TODO (CC tbd)
# 1. Virtual midi ports: for sending MIDI message between programs on same system
Relevant for:
- open_input_port()
# 2. Additional MIDI Streams: Saving/exporting MIDI files, playback from MIDI
Relevant for:
- MIDIStream (atm never instantiated)
- PlayMidiProcess (atm never instantiated)
# 3. Bugs in class FramedMidiInputProcess

"""

import time
import warnings
import multiprocessing
from multiprocessing import Pipe
import mido
from mido import Message
import rtmidi
from rtmidi.midiutil import open_midiinput, open_midioutput
from rtmidi.midiutil import get_api_from_environment

from accompanion import PLATFORM # find the OS of the current system

BACKEND = "mido"
if PLATFORM == "Windows":
    BACKEND = "rtmidi"

if PLATFORM not in ("Darwin", "Linux", "Windows"):
    warnings.warn(f"{PLATFORM} is not supported!")

# Default polling period (in seconds)
POLLING_PERIOD = 0.02


def ensure_port_info(port_id): # same
    """
    Ensure a valid (open) MIDI port
    
    Parameters
    ----------
    port_id : int or str
        Name or index of the input MIDI device
    
    Returns
    -------
    port_idx : int
        Index of the port
    port_name : str
        Name of the MIDI device
    """

    port_names = get_port_names()

    if isinstance(port_id, int):
        port_idx = port_id

        if port_idx not in list(range(len(port_names))):
            raise ValueError(f"Port index {port_idx} not available")

        port_name = port_names[port_idx]

    elif isinstance(port_id, str):
        if port_id not in port_names:
            raise ValueError(f"Port {port_id} not available")
        port_idx = port_names.index(port_id)
        port_name = port_id

    return port_idx, port_name


def get_port_names(backend=BACKEND): # same
    """
    Get MIDI port names, depending on the platform.

    Parameters
    ------
    backend : {"rtmidi", "mido"}
        platform-specific package for MIDI backend

    Returns
    -------
    port_names : list
        List of available MIDI input ports.
    """

    if backend not in ("rtmidi", "mido"):
        raise ValueError("`backend` must be in {'rtmidi', 'mido'} " f"but given {backend}"
        )

    if backend == "rtmidi":
        midi_in = rtmidi.MidiIn(get_api_from_environment(rtmidi.API_UNSPECIFIED))
        port_names = midi_in.get_ports()
        del midi_in
    elif backend == "mido":
        port_names = mido.get_input_names()

    return list(port_names)


def get_port_id(): # same, never called
    """
    Ask user to select a MIDI port.

    Returns
    -------
    port_idx : int
        Index of the port
    port_name : str
        Name of the MIDI device
    """
    port_names = get_port_names()

    if len(port_names) == 0:
        raise ValueError("No MIDI devices connected")

    print("Enter the port index displayed to the left of the MIDI device")

    for p_ix, pn in enumerate(port_names):
        print(f"{p_ix}:\t{pn}")

    # Get port index and ensure that it an acceptable option
    try:
        port_idx = int(input())
    except ValueError:
        raise ValueError("The port index must be an integer")

    if port_idx < 0 or port_idx >= len(port_names):

        raise ValueError("The port index must be in the range of 0 " f"to {len(port_names)-1}")

    return port_idx, port_names[port_idx]

 
def open_input_port(port_id, backend=BACKEND): # same
    """
    Open MIDI input port

    Parameters
    ----------
    port_id : int or str
        Name or index of the input MIDI device

    Returns
    -------
    port : mido/rtmidi supported MidiIn instance
        Created input port
    """
    # Get correct port_id depending on the platform
    port_idx, port_name = ensure_port_info(port_id)

    if backend == "rtmidi": # windows
        port, _ = open_midiinput(port_idx)

    elif backend == "mido":
        port = mido.open_input(port_name)

    return port


def open_output_port(port_id, backend=BACKEND): # same
    """
    Open MIDI output port

    Parameters
    ----------
    port_id : int or str
        Name or index of the input MIDI device

    Returns
    -------
    port : mido/rtmidi supported MidiOut instance
        Created output port
        
    TODO CC
    ----
    * verify rtmidi code

    """
    # Get correct port_id depending on the platform
    port_idx, port_name = ensure_port_info(port_id)

    if backend == "rtmidi": # windows
        port, _ = open_midioutput(port_idx)
    elif backend == "mido":
        port = mido.open_output(port_name)

    else:
        raise NotImplementedError

    return port


class MIDIStream(object): # same, never instantiated
    """
    Base class for creating MIDI Stream objects
    """

    def __init__(self, port_id, backend=BACKEND):

        self.backend = backend
        self.port_idx, self.port_name = ensure_port_info(port_id)

        self.midi_in = open_input_port(port_id, backend=backend)
        self.init_time = None

        if self.backend == "rtmidi":
            self.poll = self.poll_rtmidi
        elif self.backend == "mido":
            self.poll = self.poll_mido

    def poll_rtmidi(self):
        """
        Poll MIDI messages with RTMIDI
        """
        if self.init_time is None:
            self.init_time = time.time()

        try:
            while True:
                msg = self.midi_in.get_message()
                time.sleep(1e-6) # suspend execution b/c rtmidi seems more computationally expensive
                if msg:
                    msg_bytes, _ = msg
                    # creat mido Message so that both polling methods
                    # have the same output
                    out_msg = Message.from_bytes(msg_bytes)
                    yield out_msg, time.time() - self.init_time

        except KeyboardInterrupt:
            print("Closing port")
        finally:
            self.midi_in.close_port()

    def poll_mido(self):
        """
        Poll MIDI messages with Mido
        """
        if self.init_time is None:
            self.init_time = time.time()

        try:
            for msg in self.midi_in:
                # time.sleep(1e-6)
                yield (msg, time.time() - self.init_time)
        except KeyboardInterrupt:
            print("Closing port")

        finally:
            self.midi_in.close()


def dummy_pipeline(inputs): # same, never called
    return inputs


class MidiInputProcess(multiprocessing.Process): # same, never instantiated
    def __init__(
        self,
        port_id,
        pipe,
        backend=BACKEND,
        init_time=None,
        pipeline=None,
        return_midi_messages=False):
        multiprocessing.Process.__init__(self)

        if backend not in ("rtmidi", "mido"):
            raise ValueError("`backend` must be in {'rtmidi', 'mido'} " f"but given {backend}")

        self.backend = backend
        self.port_idx, self.port_name = ensure_port_info(port_id)
        self.init_time = init_time
        self.listen = False
        self.pipe = pipe
        self.pipeline = pipeline
        self.first_msg = False
        if pipeline is None:
            self.pipeline = dummy_pipeline
        self.return_midi_messages = return_midi_messages

        if self.backend == "mido":
            self.poll = self.poll_mido
        elif self.backend == "rtmidi":
            self.poll = self.poll_rtmidi

        self.lock = multiprocessing.RLock()

        self.midi_in = None

    def poll_rtmidi(self):
        msg = self.midi_in.get_message()
        # makes polling less cpu intensive
        time.sleep(1e-6)
        if msg:
            msg_bytes, _ = msg
            # creat mido Message so that both polling methods
            # have the same output
            msg = Message.from_bytes(msg_bytes)
        return msg

    def poll_mido(self):
        return self.midi_in.poll()

    def run(self):
        self.start_listening()

        while self.listen:
            msg = self.poll()
            if msg is not None:
                c_time = self.current_time
                # To have the same output as other MidiThreads
                output = self.pipeline([(msg, c_time)], c_time)

                if self.return_midi_messages:
                    self.pipe.send((msg, c_time), output)
                else:
                    self.pipe.send(output)

    @property
    def current_time(self):
        """
        Get current time since starting to listen
        """
        return time.time() - self.init_time

    def start_listening(self):
        """
        Start listening to midi input (open input port and
        get starting time)
        """
        self.midi_in = open_input_port(self.port_idx, self.backend)

        self.listen = True
        if self.init_time is None:
            self.init_time = time.time()

    def stop_listening(self):
        """
        Stop listening to MIDI input
        """
        # break while loop in self.run
        with self.lock:
            self.listen = False
            # reset init time

            if self.midi_in is not None:
                self.init_time = None
                if self.backend == "mido":
                    self.midi_in.close()
                elif self.backend == "rtmidi":
                    self.midi_in.close_port()
                self.midi_in = None
                print("Closing port")
        # close port
        # Join thread
        self.join()


class PlayMidiProcess(multiprocessing.Process): # same, never instantiated
    def __init__(self, port_id, filename, backend=BACKEND):
        multiprocessing.Process.__init__(self)

        if backend not in ("rtmidi", "mido"):
            raise ValueError(
                "`backend` must be in {'rtmidi', 'mido'} " f"but given {backend}"
            )

        self.backend = backend
        self.port_idx, self.port_name = ensure_port_info(port_id)

        self.continue_playing = True
        self.first_msg = False
        self.midi_out = open_output_port(self.port_idx, self.backend)

        if self.backend == "mido":
            self.mid = mido.MidiFile(filename)
            self.play = self.play_mido
        elif self.backend == "rtmidi":
            # TODO open midi file
            self.send = self.play_rtmidi

    def play_rtmidi(self):
        # TODO implement
        raise NotImplementedError

    def play_mido(self):
        for msg in self.mid.play():

            if not self.continue_playing:
                break

            self.midi_out.send(msg)

    def run(self):
        self.play()

    def stop_playing(self):
        """
        Stop listening to MIDI input
        """
        # break while loop in self.run
        self.continue_playing = False

        # close port
        if self.backend == "mido":
            self.midi_out.close()
        elif self.backend == "rtmidi":
            self.midi_out.close_port()
        # Join thread
        self.join()
        print("Closing output port")


class Buffer(object): # same, only called by FrameMidiInputProcess
    """
    Base class for creating MIDI buffers to separately handle MIDI Messages
    """
    def __init__(self, polling_period):
        self.polling_period = polling_period
        self.frame = []
        self.start = None

    def append(self, input, time):
        self.frame.append((input, time))

    def reset(self, time):
        self.frame = []
        self.start = time

    @property
    def end(self):
        """
        Maximal end time of the frame
        """
        return self.start + self.polling_period

    @property
    def time(self):
        """
        Middle of the frame
        """
        return self.start + 0.5 * self.polling_period

    def __len__(self):
        return len(self.frame)

    def __str__(self):
        return str(self.frame)


class FramedMidiInputProcess(MidiInputProcess): # same, only called by create_poll_process_midi() which is never called itself
    """
    Base class for creating MIDI Messages with frame data
    """
    def __init__(
        self,
        port_id,
        pipe,
        polling_period=POLLING_PERIOD,
        backend=BACKEND,
        init_time=None,
        pipeline=None,
        return_midi_messages=False,
    ):
        super().__init__(
            port_id=port_id,
            pipe=pipe,
            backend=backend,
            init_time=init_time,
            pipeline=pipeline,
            return_midi_messages=return_midi_messages,
        )
        self.polling_period = polling_period

    def run(self):
        """
        TODO
        ----
        * Fix Error with c_time when stopping the thread
        * Adapt sleep time from midi_online
        """
        sttime = time.time()
        self.start_listening()
        frame = Buffer(self.polling_period)
        frame.start = self.current_time
        # TODO: Adapt from midi_online to allow for variable polling
        # periods?
        while self.listen:
            # added if to check once again after sleep
            # TODO verify if still correct
            c_time = self.current_time
            msg = self.poll()
            if msg is not None:
                frame.append(msg, c_time)
                if not self.first_msg:
                    self.first_msg = True
            if c_time >= frame.end and self.first_msg:
                output = self.pipeline((frame.frame, c_time))
                # self.pipe.send(output)
                if self.return_midi_messages:
                    self.pipe.send((frame.frame, output))
                else:
                    self.pipe.send(output)
                frame.reset(c_time)


def create_poll_process_midi(port_id, polling_period, pipeline, return_midi_messages=False, backend=BACKEND): # same, never called
    """
    Helper to create a FramedMidiInputProcess and its respective pipe.
    """
    p_output, p_input = Pipe()
    mt = FramedMidiInputProcess(
        port_id=port_id,
        pipe=p_output,
        polling_period=polling_period,
        backend=backend,
        pipeline=pipeline,
        return_midi_messages=return_midi_messages,
    )
    return p_output, p_input, mt
