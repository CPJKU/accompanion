# -*- coding: utf-8 -*-
"""
This module provides basic functionality to process MIDI inputs in
real time. This is a copy from matchmaker/io/midi.py, so that it can
be updated without requiring to re-install matchmaker
"""
import time
import datetime
import queue
# import sys

import mido

from accompanion.midi_handler.fluid import FluidsynthPlayer
from accompanion.midi_handler.midi_utils import (
    midi_file_from_midi_msg,
    OUTPUT_MIDI_FOLDER,
)
import os


class MidiRouter(object):
    """
    This is the main class handling MIDI I/O.
    It takes (partial) strings for port names as inputs
    and searches for a fitting port.
    The reason this is set up in this way is that
    different OS tend to name/index MIDI ports differently.

    Use an instance if this class (and *only* this instance)
    to handle everything related to port opening, closing,
    finding, and panic. Expecially Windows is very finicky
    about MIDI ports and it'll likely break if ports are
    handled separately.

    This class can be used to:
    - create a midirouter = MidiRouter(**kwargs) with
    a number of (partial) port names or fluidsynths
    - poll a specific port: e.g.
    midirouter.solo_input_to_accompaniment_port.poll()
    - send on a specific port: e.g.
    midirouter.acc_output_to_sound_port.send(msg)
    - open all set ports: midirouter.open_ports()
    - close all set ports: midirouter.close_ports()
    - panic reset all ports: midirouter.panic()
    - get the full name of the used midi ports: e.g.
    midirouter.solo_input_to_accompaniment_port_name
    (DON'T use this name to open, close, etc with it,
    use the midirouter functions instead)

    Args:
        solo_input_to_accompaniment_port_name (string):
            a (partial) string for the input name at which the
            score follower is listening for soloist MIDI messages
        acc_output_to_sound_port_name (string):
            a (partial) string for the output name where the
            accompanist sends MIDI messages
            alternatively, it takes a FluidSynthPlayer object,
            any out messages are then sent to a fludisynth for
            audio rendering.

    It is possible to use a built-in MIDIPlayer instead of a soloist.
    The MIDIPlayer sends midi messages to MIDI and/or sound ports.
    Note that a virtual MIDI connection is necessary to send messages
    froom the MIDIPlayer to the Score Follower. Virtual MIDI connections
    are available on MacOSX (via IAC Driver) and Windows (
    https://www.tobias-erichsen.de/software/loopmidi.html
    ).

    MIDIPlayer_to_sound_port_name (string):
        a (partial) string for the output name where the
        MIDIPlayer sends MIDI messages
        alternatively, it takes a FluidSynthPlayer object,
        any out messages are then sent to a fludisynth for
        audio rendering.
    MIDIPlayer_to_accompaniment_port_name (string):
        a (partial) string for the output name where the
        MIDIPlayer sends MIDI messages. Most likely virtual
        port that loops back to the accompanion input.


    It is possible to use a built-in controllable MIDIPlayer
    (which plays a midi file based in a single button)
    to play as a soloist.
    Use any midi controller as input to this Player.

    simple_button_input_port_name (string):
        a (partial) string for the input name at which the
        MIDIPlayer is listening for
        "single button player" MIDI messages.
    """

    def __init__(
        self,
        solo_input_to_accompaniment_port_name=None,
        acc_output_to_sound_port_name=None,
        MIDIPlayer_to_sound_port_name=None,
        MIDIPlayer_to_accompaniment_port_name=None,
        simple_button_input_port_name=None,
    ):
        self.available_input_ports = mido.get_input_names()
        print("Available inputs MIDI for mido", self.available_input_ports)
        self.available_output_ports = mido.get_output_names()
        print("Available outputs MIDI for mido", self.available_output_ports)
        # try:
        #     self.available_input_ports = mido.get_input_names()
        #     self.available_output_ports = mido.get_output_names()
        #     print("Available outputs MIDI for mido", self.available_output_ports)
        # except RuntimeError as e:
        #     print(e)
        #     print("No ports available, mido crashes, switching to dummy input ports.")
        #     self.available_input_ports = []
        #     self.available_output_ports = []
        self.input_port_names = {}
        self.output_port_names = {}
        self.open_ports_list = []

        # the MIDI port name the accompanion listens at (port name)
        self.solo_input_to_accompaniment_port_name = self.proper_port_name(
            solo_input_to_accompaniment_port_name
        )
        # the MIDI port name / Instrument the accompanion is sent
        # to (Fluidsynth, port name)
        self.acc_output_to_sound_port_name = self.proper_port_name(
            acc_output_to_sound_port_name, False
        )

        # the MIDI port name / Instrument (if any) the solo is sent to,
        # if a MIDI Player is used (Fluidsynth, port name, None)
        self.MIDIPlayer_to_sound_port_name = self.proper_port_name(
            MIDIPlayer_to_sound_port_name, False
        )
        # the MIDI port name (if any) the solo is sent for the accompanion
        # to listen, if a MIDI Player is used (port name, None)
        self.MIDIPlayer_to_accompaniment_port_name = self.proper_port_name(
            MIDIPlayer_to_accompaniment_port_name, False
        )
        # the MIDI port name (if any) a single button MIDI Player is listening
        # at (port name, None)
        self.simple_button_input_port_name = self.proper_port_name(
            simple_button_input_port_name, True
        )

        self.open_ports()

        self.solo_input_to_accompaniment_port = self.assign_ports_by_name(
            self.solo_input_to_accompaniment_port_name, input=True
        )
        self.acc_output_to_sound_port = self.assign_ports_by_name(
            self.acc_output_to_sound_port_name, input=False
        )

        self.MIDIPlayer_to_sound_port = self.assign_ports_by_name(
            self.MIDIPlayer_to_sound_port_name, input=False
        )
        self.MIDIPlayer_to_accompaniment_port = self.assign_ports_by_name(
            self.MIDIPlayer_to_accompaniment_port_name, input=False
        )
        self.simple_button_input_port = self.assign_ports_by_name(
            self.simple_button_input_port_name
        )

        self.MIDIPlayer_port = self.assign_midi_player_out()

    def proper_port_name(self, try_name, input=True):
        ## TODO: Simplify using version from matchmaker
        if isinstance(try_name, str):
            if input:
                possible_names = [
                    (i, name)
                    for i, name in enumerate(self.available_input_ports)
                    if try_name in name
                ]
            else:
                possible_names = [
                    (i, name)
                    for i, name in enumerate(self.available_output_ports)
                    if try_name in name
                ]
                # possible_names = ["RD-88 1"]
                # print("Possibly output names", possible_names)
                # print("Try name", try_name)

            if len(possible_names) == 1:
                print(
                    "port name found for trial name: ",
                    try_name,
                    "the port is set to: ",
                    possible_names[0],
                )
                if input:
                    self.input_port_names[possible_names[0][1]] = None
                else:
                    self.output_port_names[possible_names[0][1]] = None
                return possible_names[0]

            elif len(possible_names) < 1:
                print("no port names found for trial name: ", try_name)
                return None
            elif len(possible_names) > 1:
                print(" many port names found for trial name: ", try_name)
                if input:
                    self.input_port_names[possible_names[0][1]] = None
                else:
                    self.output_port_names[possible_names[0][1]] = None
                return possible_names[0]
                # return None
        elif isinstance(try_name, int):
            if input:
                try:
                    possible_name = (try_name, self.available_input_ports[try_name])
                    self.input_port_names[possible_name[1]] = None
                    return possible_name
                except ValueError:
                    raise ValueError(f"no input port found for index: {try_name}")
            else:
                try:
                    possible_name = (try_name, self.available_output_ports[try_name])
                    self.output_port_names[possible_name[1]] = None
                    return possible_name
                except ValueError:
                    raise ValueError(f"no output port found for index: {try_name}")

        elif isinstance(try_name, FluidsynthPlayer):
            return try_name

        else:
            return None

    def open_ports_by_name(self, try_name, input=True):
        if try_name is not None:
            if input:
                port = mido.open_input(try_name)
            else:
                port = mido.open_output(try_name)
                # Adding eventual key release.
                port.reset()

            self.open_ports_list.append(port)
            return port

        else:
            return try_name

    def open_ports(self):
        for port_name in self.input_port_names.keys():
            if self.input_port_names[port_name] == None:
                port = self.open_ports_by_name(port_name, input=True)
                self.input_port_names[port_name] = port
        for port_name in self.output_port_names.keys():
            if self.output_port_names[port_name] == None:
                port = self.open_ports_by_name(port_name, input=False)
                self.output_port_names[port_name] = port

    def close_ports(self):
        for port in self.open_ports_list:
            port.close()
        self.open_ports_list = []

        for port_name in self.output_port_names.keys():
            self.output_port_names[port_name] = None
        for port_name in self.input_port_names.keys():
            self.input_port_names[port_name] = None

    def panic(self):
        for port in self.open_ports_list:
            port.panic()

    def assign_ports_by_name(self, try_name, input=True):
        if isinstance(try_name, FluidsynthPlayer):
            return try_name
        elif try_name is not None:
            if input:
                return self.input_port_names[try_name[1]]
            else:
                return self.output_port_names[try_name[1]]
        else:
            return None

    def assign_midi_player_out(self):
        if (
            self.MIDIPlayer_to_sound_port is not None
            and self.MIDIPlayer_to_accompaniment_port is not None
        ):
            # if isinstance(self.MIDIPlayer_to_sound_port, FluidsynthPlayer):
            return DummyMultiPort(
                self.MIDIPlayer_to_accompaniment_port, self.MIDIPlayer_to_sound_port
            )
            # else:
            #     return mido.MultiPort([self.MIDIPlayer_to_sound_port, self.MIDIPlayer_to_accompaniment_port])
        elif self.MIDIPlayer_to_accompaniment_port is not None:
            return self.MIDIPlayer_to_accompaniment_port
        elif self.MIDIPlayer_to_sound_port is not None:
            return self.MIDIPlayer_to_sound_port
        else:
            return None


class DummyMultiPort(object):
    def __init__(self, midi_port, fluid_port):
        self.midi_port = midi_port
        self.fluid_port = fluid_port

    def send(self, msg):
        self.midi_port.send(msg)
        self.fluid_port.send(msg)


class DummyPort(object):
    def __init__(self, *args, **kwargs):
        pass

    def send(self, msg):
        pass

    def panic(self):
        pass

    def poll(self):
        pass

    def reset(self):
        pass


class MidiFilePlayerInterceptPort(object):
    def __init__(self, *args, **kwargs):
        self.queue = queue.Queue()
        self.active = True

    def send(self, msg):
        self.queue.put(msg)

    def panic(self):
        self.active = False

    def poll(self):
        while self.active:
            try:
                msg = self.queue.get(True, 1)
                return msg
            except queue.Empty:
                pass


class DummyRouter(object):
    """"""

    def __init__(
        self,
        solo_input_to_accompaniment_port_name=None,
        acc_output_to_sound_port_name=None,
        MIDIPlayer_to_sound_port_name=None,
        MIDIPlayer_to_accompaniment_port_name=None,
        simple_button_input_port_name=None,
    ):
        self.available_input_ports = None
        self.available_output_ports = None
        print("Available outputs MIDI for mido", self.available_output_ports)
        self.input_port_names = {}
        self.output_port_names = {}
        self.open_ports_list = []

        # the MIDI port name the accompanion listens at (port name)
        self.solo_input_to_accompaniment_port_name = None

        self.solo_input_to_accompaniment_port = MidiFilePlayerInterceptPort()

        # the MIDI port name / Instrument the accompanion is sent
        # to (Fluidsynth, port name)
        self.acc_output_to_sound_port_name = None

        # the MIDI port name / Instrument (if any) the solo is sent to,
        # if a MIDI Player is used (Fluidsynth, port name, None)
        self.MIDIPlayer_to_sound_port_name = None
        # the MIDI port name (if any) the solo is sent for the accompanion
        # to listen, if a MIDI Player is used (port name, None)

        self.MIDIPlayer_to_accompaniment_port_name = None
        # the MIDI port name (if any) a single button MIDI Player is listening
        # at (port name, None)
        self.simple_button_input_port_name = None

        self.open_ports()

        self.acc_output_to_sound_port = self.assign_ports_by_name(
            self.acc_output_to_sound_port_name, input=False
        )

        self.MIDIPlayer_to_sound_port = self.assign_ports_by_name(
            self.MIDIPlayer_to_sound_port_name, input=False
        )
        self.MIDIPlayer_to_accompaniment_port = self.solo_input_to_accompaniment_port
        self.simple_button_input_port = self.assign_ports_by_name(
            self.simple_button_input_port_name
        )

        self.MIDIPlayer_port = self.assign_midi_player_out()

    def proper_port_name(self, try_name, input=True):
        return None

    def open_ports_by_name(self, try_name, input=True):
        return DummyPort()

    def open_ports(self):
        pass

    def close_ports(self):
        self.solo_input_to_accompaniment_port.active = False

    def panic(self):
        pass

    def assign_ports_by_name(self, try_name, input=True):
        return DummyPort()

    def assign_midi_player_out(self):
        return None


class RecordingRouter(MidiRouter):
    """This class works like a standard MIDI router and in addition ot the MIDI input from
    the soloist and the MIDI output of the accompaniment.
    """

    def __init__(self, piece_name, **router_kwargs):
        super(RecordingRouter, self).__init__(**router_kwargs)
        self.piece_name = piece_name
        self.solo_input_to_accompaniment_port = RecordingPort(
            self.solo_input_to_accompaniment_port
        )
        self.acc_output_to_sound_port = RecordingPort(self.acc_output_to_sound_port)

    def close_ports(self):
        super(RecordingRouter, self).close_ports()
        self.save_midi()

    def save_midi(self):
        all_msg_soloits = list(self.solo_input_to_accompaniment_port.all_msg.queue)
        all_msg_acc = list(self.acc_output_to_sound_port.all_msg.queue)
        # The date format is {Year}-{Month}-{Day}_{Hours}-{Minutes}-{Seconds}
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # save input soloist
        soloist_out_path = os.path.join(
            OUTPUT_MIDI_FOLDER, f"{self.piece_name}_soloist_{time_str}.mid"
        )
        midi_file_from_midi_msg(all_msg_soloits, soloist_out_path)
        # save generated accompaniment
        accompaniment_out_path = os.path.join(
            OUTPUT_MIDI_FOLDER, f"{self.piece_name}_accompaniment_{time_str}.mid"
        )
        midi_file_from_midi_msg(all_msg_acc, accompaniment_out_path)
        print(f"MIDI files saved in {soloist_out_path} and {accompaniment_out_path}")


class RecordingPort(object):
    """This class acts a middleman MIDI port for recording MIDI msgs.
    It captures messages sent and received and forward them to the wanted port
    """

    def __init__(self, real_port):
        self.active = True
        self.port = real_port
        self.all_msg = queue.Queue()

    def send(self, msg):
        if msg is not None:
            self.all_msg.put((msg, time.time()))
        self.port.send(msg)

    # def panic(self):
    #     self.active=False

    def poll(self):
        msg = self.port.poll()
        if msg is not None:
            self.all_msg.put((msg, time.time()))
        return msg

    def panic(self):
        self.port.panic()
