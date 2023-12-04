# -*- coding: utf-8 -*-
"""
This module contains a mediator for filtering out the MIDI messages 
"performed" by the ACCompanion from the input MIDI. Using the mediator
is only necessary for old instruments that do not filter the messages
automatically. So far we have only encountered this issue with the
Bösendorfer CEUS, and thus the name of the class.
"""
# The thread mediator was used in the following context (which was part
# the equivalent of our current MIDI input process/thread)
# # !FUNCTION! callback for MIDI input:
# def input_callback(message):
#     '''
#     Sends the incomming MIDI message to midi_manager. Filters it through
#     the mediator class for the case of a CEUS system.

#     Parameters
#     ----------
#     message : Mido.MidiMessage
#         The MIDI message.
#     '''
#     # First, measure the current time:
#     current_time = time.time()
#     # Check if we are using a filter lock:
#     if mediator.mediator_type == 'ceus':
#         # Check whether the message is not in the midi filter:
#         if message.type == 'note_on':
#             if not mediator.filter_check(message.note):
#                 # Get with the lock:
#                 with midi_lock:
#                     # Update the buffer for the current sample:
#                     midi_manager.addObservation(message, current_time)

#     else:
#         # Get with the lock:
#         with midi_lock:
#             # Update the buffer for the current sample:
#             midi_manager.addObservation(message, current_time)


# Native Python Library:
import collections
import threading


class ThreadMediator(object):
    """
    Mediator class for communications between ACCompanion modules running in
    concurrent threads or processes. The class ensures thread safety.

    Parameters
    ----------
    _comms_buffer : deque
        A buffer to receive the output from the one process and send it to
        another when promped. Follows LIFO (Last In, First Out) logic. For the
        buffer a deque object is used as this ensures thread safety.
    """

    _comms_buffer: collections.deque
    _mediator_type: str

    def __init__(self, **kwargs) -> None:
        """
        The initialization method.
        """
        # Define the comms buffer:
        self._comms_buffer = collections.deque(maxlen=200)
        # A name variable to store the type of the mediator:
        self._mediator_type = "default"
        # Call the superconstructor:
        super().__init__(**kwargs)

    def is_empty(self) -> bool:
        """
        Returns True if the comms buffer is empty. False if it has at least one
        element.

        Returns
        -------
        empty : Boolean
            True if the comms buffer is empty. False if it has at least one
            element.
        """
        # Check if the buffer is empty:
        if len(self._comms_buffer) == 0:
            return True
        # Otherwise return false:
        return False

    def get_message(self) -> collections.namedtuple:
        """
        Get the first from the previously sent messages from an ACCompanion
        module. Returns IndexError if there is no element in the buffer.
        This should only be called by the ACCompanion accompaniment production
        module.

        Returns
        -------
        message : collections.namedtuple
            The message to be returned.
        """
        # Return the first element:
        return self._comms_buffer.popleft()

    def put_message(self, message: collections.namedtuple) -> None:
        """
        Put a message into the comms buffer.
        This should only be called by ACCompanion's score matching/following
        module.

        Parameters
        ----------
        message : collections.namedtuple
            The message to be put into the buffer.
        """
        self._comms_buffer.append(message)

    @property
    def mediator_type(self) -> str:
        """
        Property method to return the value of the mediator_type variable.
        """
        return self._mediator_type


class CeusMediator(ThreadMediator):
    """
    Encapsulates the ACCompanion trans-module communication in the context of a
    Ceus System. It also filters notes (MIDI pitches) in the accompaniment part
    that are played by Ceus. These notes are fed back to the matcher (MAPS) and
    thus need to be filtered.

    Parameters
    ----------
    _Ceus_filter : deque
        The filter buffer for notes (MIDI pitches) played back by Ceus.

    _comms_buffer : deque
        A buffer to receive the output from the one process and send it to
        another when promped. Follows LIFO (Last In, First Out) logic. For the
        buffer a deque object is used as this ensures thread safety.
    """

    _ceus_lock: threading.RLock
    def __init__(self, **kwargs):
        """
        The initialization method.
        """
        # A lock to ensure thread safety of the Ceus filter:
        self._ceus_lock = threading.RLock()

        # Define the Ceus filter:
        self._ceus_filter = collections.deque(maxlen=10)
        # Call the superconstructor:
        super().__init__(**kwargs)

        # A name variable to store the type of the mediator:
        self._mediator_type = "ceus"

    def filter_check(self, midi_pitch: int, delete_entry: bool=True) -> bool:
        """
        Check if the midi pitch is in the Ceus filter. Return True if yes,
        False if it is not present. Delete the filter entry if specified by
        delete_entry.

        Parameters
        ----------
        midi_pitch : int
            The midi pitch to be checked against the filter.

        delete_entry : bool
            Specifies whether to delete the filter entry if such is found.
            Default: True

        Returns
        -------
        indicate : bool
            True if pitch is in the filter, False if it is not.
        """

        with self._ceus_lock:
            # print("\t", self._ceus_filter)
            # Check if the entry is in the filter:
            if midi_pitch in self._ceus_filter:

                # Whether to delete the entry:
                if delete_entry:
                    self.filter_remove_pitch(midi_pitch)

                # Return true:
                return True

            # If it is not, return False:
            return False

    def filter_append_pitch(self, midi_pitch: int) -> None:
        """
        Append a MIDI pitch to the Ceus filter. This should only be called by
        the ACCompanion's accompaniment production module.

        Parameters
        ----------
        midi_pitch : int
            The midi pitch to be appended to the filter.
        """
        with self._ceus_lock:
            # Append the midi pitch to be filtered:
            self._ceus_filter.append(midi_pitch)

    def filter_remove_pitch(self, midi_pitch: int) -> None:
        """
        Remove a MIDI pitch from the Ceus filter. This should only be called by
        the ACCompanion's score matching/following module.

        Parameters
        ----------
        midi_pitch : int
            The midi pitch to be removed from the filter.
        """
        with self._ceus_lock:
            # Remove the pitch from filter:
            self._ceus_filter.remove(midi_pitch)
