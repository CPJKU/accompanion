# -*- coding: utf-8 -*-
"""
This module provides basic functionality to process MIDI inputs in
real time. This is a copy from matchmaker/io/midi.py, so that it can
be updated without requiring to re-install matchmaker
"""
import datetime
import os
import queue
import time
from typing import Iterable, Optional, Union

import mido
from mido.ports import BaseOutput

from accompanion.midi_handler.fluid import FluidsynthPlayer
from accompanion.midi_handler.midi_utils import (
    OUTPUT_MIDI_FOLDER,
    RECORDED_TYPES,
    close_sounding_notes,
    first_note_time,
    write_midi,
)

# import sys


class BasePort(object):
    """
    Base class for custom ports. All custom ports need
    to implement at least the `send` method.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def send(self, msg: mido.Message) -> None:
        """
        Send a MIDI message

        Parameters
        ----------
        msg: mido.Message
           MIDI Message to be sent through the port.
        """
        raise NotImplementedError

    def panic(self) -> None:
        """
        Panic button to stop all MIDI notes.
        """
        pass

    def poll(self) -> Optional[mido.Message]:
        """
        Poll message from the port. Needs to be implemented
        by the subclasses if they serve as input ports.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset all activity in the MIDI port (stop all notes, programs and controllers).
        """
        pass


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
            if self.input_port_names[port_name] is None:
                port = self.open_ports_by_name(port_name, input=True)
                self.input_port_names[port_name] = port
        for port_name in self.output_port_names.keys():
            if self.output_port_names[port_name] is None:
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


class DummyMultiPort(BasePort):
    def __init__(self, midi_port, fluid_port):
        super().__init__()
        self.midi_port = midi_port
        self.fluid_port = fluid_port

    def send(self, msg):
        self.midi_port.send(msg)
        self.fluid_port.send(msg)


class DummyPort(BasePort):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def send(self, msg):
        pass

    def poll(self):
        pass


class MultiOutputPort(BasePort):
    """
    Virtual MIDI port that sends messages to multiple output ports.

    Parameters
    ----------
    output_ports: BaseOutput, BasePort or iterable of those classes
        Output ports that we want to send the same messages to.
    """

    def __init__(
        self,
        output_ports: Union[Iterable[BaseOutput], BaseOutput, BasePort],
    ) -> None:

        if isinstance(output_ports, (BaseOutput, BasePort)):
            self.output_ports = [output_ports]

        elif isinstance(output_ports, Iterable):
            if any(not isinstance(op) for op in output_ports):
                raise ValueError(
                    "All provided output ports should be of type " "`BaseOutput`!"
                )
            self.output_ports = output_ports
        else:
            raise ValueError(
                "`output_ports` should be a ` BaseOutput` instance"
                f"or a list of instances, but it is {type(output_ports)}."
            )

    def send(self, msg):

        for port in self.output_ports:
            port.send(msg)

    def panic(self):

        for port in self.output_ports:
            port.panic()

    def reset(self):

        for port in self.output_ports:
            port.reset()


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


class MidiRecorder(object):
    """Records the soloist's input and the accompaniment's output of a session.

    Both parts are timestamped against one origin -- by default the same
    `time.time()` the sequencer and the MIDI input count from -- so the two
    files describe a single performance rather than two takes, and can be laid
    over each other without alignment.

    The two ports are wrapped in place on whatever router is in use, so this
    works for a real `MidiRouter` and equally for the `DummyRouter` that
    `test=True` installs: a silent MIDI-file rehearsal is recorded just as a
    played one is.

    Parameters
    ----------
    output : str
        Directory to write the session into. Created if missing; an existing
        recording in it is overwritten.
    origin : float, optional
        Time zero for the recording, as `time.time()` reads it. Defaults to
        when the recorder was made.
    """

    #: ``(file stem, track name, router attribute)`` per recorded part.
    PARTS = (
        ("soloist", "Soloist", "solo_input_to_accompaniment_port"),
        ("accompaniment", "Accompaniment", "acc_output_to_sound_port"),
    )

    def __init__(self, output, origin=None):
        self.output = output
        self.origin = time.time() if origin is None else float(origin)
        self.end = None
        self.ports = {}

    def attach(self, router):
        """Wrap the router's soloist input and accompaniment output.

        Call before the sequencer and the MIDI input are built, since both
        take their port from the router once and keep it.
        """
        for stem, _, attribute in self.PARTS:
            port = RecordingPort(getattr(router, attribute), self.origin)
            setattr(router, attribute, port)
            self.ports[stem] = port

    def stop(self):
        """Mark the end of the performance, before the ports are torn down."""
        if self.end is None:
            self.end = time.time() - self.origin

    def messages(self, stem):
        """One part's captured messages, with anything left sounding released."""
        port = self.ports.get(stem)
        captured = list(port.messages) if port is not None else []
        end = max([self.end or 0.0] + [seconds for _, seconds in captured])
        return close_sounding_notes(captured, end)

    def save(self):
        """Write each part, both parts together, and their shared offsets.

        Every file is in session seconds, so the parts already line up. The
        offsets name each part's first note on that same timeline, for a
        reader that wants to place them itself.
        """
        self.stop()
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        parts = {stem: self.messages(stem) for stem, _, _ in self.PARTS}
        for stem, name, _ in self.PARTS:
            write_midi(os.path.join(self.output, f"{stem}.mid"), [(name, parts[stem])])
        write_midi(
            os.path.join(self.output, "duo.mid"),
            [(name, parts[stem]) for stem, name, _ in self.PARTS],
        )
        onsets = {stem: first_note_time(parts[stem]) for stem, _, _ in self.PARTS}
        with open(os.path.join(self.output, "sync.csv"), "w") as stream:
            stream.write("filename,offset_s\n")
            for stem, _, _ in self.PARTS:
                onset = onsets[stem]
                # A part that never sounded has no first note to place, and an
                # unknown offset is left empty rather than written as zero.
                value = "" if onset is None else f"{onset:.6f}"
                stream.write(f"{stem}.mid,{value}\n")
        counts = ", ".join(
            "{} {} notes".format(
                stem,
                sum(
                    message.type == "note_on" and message.velocity > 0
                    for message, _ in parts[stem]
                ),
            )
            for stem, _, _ in self.PARTS
        )
        print(f"Recorded {counts} in {self.output}")
        return self.output


class RecordingRouter(MidiRouter):
    """A MIDI router that also records what the soloist and accompaniment play.

    Kept for callers that ask for a recording router by name. New code can
    wrap any router -- including the `DummyRouter` used for silent rehearsal,
    which this one cannot be -- with `MidiRecorder.attach`.
    """

    def __init__(self, piece_name, output=None, origin=None, **router_kwargs):
        super(RecordingRouter, self).__init__(**router_kwargs)
        self.piece_name = piece_name
        if output is None:
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output = os.path.join(OUTPUT_MIDI_FOLDER, f"{piece_name}_{stamp}")
        self.recorder = MidiRecorder(output, origin)
        self.recorder.attach(self)

    def close_ports(self):
        # Before the ports go, so the silencing note offs the panic button
        # sends are not taken for part of the performance.
        self.recorder.stop()
        super(RecordingRouter, self).close_ports()
        self.save_midi()

    def save_midi(self):
        return self.recorder.save()


class RecordingPort(BasePort):
    """A MIDI port that timestamps everything passing through it.

    One upstream port is wrapped and every other attribute is delegated -- read
    *and* written, so that a router closing `active` on the port it handed out
    still reaches the real one. Times are seconds since the recorder's origin.
    """

    #: Everything else belongs to the wrapped port.
    _OWN = frozenset({"port", "origin", "messages"})

    def __init__(self, real_port, origin=None):
        super().__init__()
        object.__setattr__(self, "port", real_port)
        object.__setattr__(self, "origin", time.time() if origin is None else origin)
        object.__setattr__(self, "messages", [])

    def _capture(self, msg):
        if msg is not None and msg.type in RECORDED_TYPES:
            # The sequencer reuses its note messages, so keep a copy.
            self.messages.append((msg.copy(), time.time() - self.origin))
        return msg

    def send(self, msg):
        self.port.send(self._capture(msg))

    def poll(self):
        return self._capture(self.port.poll())

    def panic(self):
        self.port.panic()

    @property
    def all_msg(self):
        """The captured messages, as the queue older callers expect."""
        held = queue.Queue()
        for item in self.messages:
            held.put(item)
        return held

    def __getattr__(self, name):
        return getattr(self.port, name)

    def __setattr__(self, name, value):
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.port, name, value)
