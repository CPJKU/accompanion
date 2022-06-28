# -*- coding: utf-8 -*-
"""
This module provides basic functionality to process MIDI inputs in
real time. This is a copy from matchmaker/io/midi.py, so that it can
be updated without requiring to re-install matchmaker
"""

import mido

# platform checking
import platform
import warnings

PLATFORM = platform.system()

if PLATFORM not in ("Darwin", "Linux", "Windows"):
    warnings.warn(f"{PLATFORM} is not supported!")

if PLATFORM == "Linux":
    MIDI_DRIVER = "alsa"
elif PLATFORM == "Darwin":
    MIDI_DRIVER = "coreaudio"
elif PLATFORM == "Windows":
    print("Experimental Windows support")
    MIDI_DRIVER = None
    # raise ValueError("Windows is not supported yet...")


if PLATFORM != "Windows":
    from accompanion.fluid import FluidsynthPlayer
else:
    from accompanion.fluid import FluidsynthPlayerWindows as FluidsynthPlayer
    # unfortunately, this import breaks if fluidsynth is not installed
    # class FluidsynthPlayer(object):
    #     def __init__(self):
    #         self.name = "FluidsynthPlayer dummy"

    #     def send(self, message):
    #         warnings.warn(
    #             f"Fluidsynth Player is currently not supported under "
    #             "windows, use a MIDI output port!"
    #         )   

class MidiRouter(object):
    def __init__(
        self,
        solo_input_to_accompaniment_port_name=None,
        acc_output_to_sound_port_name=None,
        MIDIPlayer_to_sound_port_name=None,
        MIDIPlayer_to_accompaniment_port_name=None,
        simple_button_input_port_name=None,
    ):
        self.available_input_ports = mido.get_input_names()
        self.available_output_ports = mido.get_output_names()
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
                    possible_name = (
                        try_name,
                        self.available_input_ports[try_name]
                    )
                    self.input_port_names[possible_name[1]] = None
                    return possible_name
                except ValueError:
                    raise ValueError(
                        f"no input port found for index: {try_name}"
                        )
            else:
                try:
                    possible_name = (
                        try_name,
                        self.available_output_ports[try_name]
                    )
                    self.output_port_names[possible_name[1]] = None
                    return possible_name
                except ValueError:
                    raise ValueError(
                        f"no output port found for index: {try_name}"
                    )

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

            self.open_ports_list.append(port)
            return port

        else:
            return try_name

    def open_ports(self):
        for port_name in self.input_port_names.keys():
            port = self.open_ports_by_name(port_name, input=True)
            self.input_port_names[port_name] = port
        for port_name in self.output_port_names.keys():
            port = self.open_ports_by_name(port_name, input=False)
            self.output_port_names[port_name] = port

    def close_ports(self):
        for port in self.open_ports_list:
            port.close()
        self.open_ports_list = []

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
                self.MIDIPlayer_to_accompaniment_port,
                self.MIDIPlayer_to_sound_port
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
