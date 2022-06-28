import mido
import multiprocessing
import time

from rtmidi.midiutil import open_midiinput, open_midioutput


MIDI_KEYS = 88

# change according to MIDI keyboard
# NOTE_ON = 158
# NOTE_OFF = 142

# for foldable MIDI keyboard at CP
NOTE_ON = 144
NOTE_OFF = 128


def read_midi(port, filename):
    mid = mido.MidiFile(filename)
    midi_out, port_name = open_midioutput(port)

    try:
        for msg in mid.play():

            if hasattr(msg, "time"):
                time.sleep(msg.time)

            event = None
            if msg.type == 'note_on':
                event = [NOTE_ON, msg.note, msg.velocity]

            if msg.type == 'note_off':
                event = [NOTE_OFF, msg.note, msg.velocity]

            if event is not None:
                midi_out.send_message(event)

    finally:
        midi_out.close_port()
        print('Close playing port')


# adapted from Nimrod
def group_chords_exactly(frame, remainder=[]):
    if len(frame) == 0:
        return [], []

    frames = [[frame[0]]]

    for x in frame[1:]:
        _, dt = x

        if dt == 0:
            frames[-1].append(x)
        else:
            frames.append([x])

    return frames, []


def default_preprocess(f, r):
    return [f], []


def create_poll_process_midi(port_id, init_polling_period,preprocess=default_preprocess):
    poll_conn_read, poll_conn_write = multiprocessing.Pipe()


    poll_process = multiprocessing.Process(target=period_poll_midi,args=(port_id, poll_conn_write,
                                                                         init_polling_period, preprocess))

    #	poll_conn_read is used to read the MIDI frames
    #	poll_process is a process handle and needs to be started by the user since starting time is up to the user
    #   (joining or closing process is not really necessary (this process can be killed via KeyboardInterrupt anyway),
    #   but should be done on platforms which do not do automatic resource clean up on program end)
    return poll_conn_read, poll_process


# TODO: Check if the exact window size issue can be solved via MIDI clock messages
def period_poll_midi(port_id, output_connection, init_polling_period, preprocess, polling_period_connection=None):
    # NOTE: 	preprocess can be used to filter and group subframes of the total frame that is polled periodically
    #		    see group_chords_exactly above for an example implementation

    try:
        midi_in, port_name = open_midiinput(port_id)
    except:
        print(f"Couldn't open MIDI port {port_id}")
        output_connection.close()
        if polling_period_connection:
            polling_period_connection.close()
        return

    def connection_wrapper(polling_period):
        try:
            if polling_period_connection.poll():
                return polling_period_connection.recv()
        except:
            print("WARNING: Poll period cannot be updated")

        return polling_period

    if polling_period_connection:
        update_polling_period = connection_wrapper

    else:
        update_polling_period = lambda x: x

    try:
        polling_period = init_polling_period
        remainder = []
        frame = []

        total_t = polling_period + time.perf_counter()

        dt_correction = 0
        while True:
            st = total_t - time.perf_counter()

            if st > 0:
                time.sleep(st)

            total_t = polling_period + time.perf_counter()

            len_correction = 0
            msg = midi_in.get_message()

            if msg:
                frame.append(msg)
                _, dt = msg
                t = dt - dt_correction

                while True:
                    msg = midi_in.get_message()

                    if msg:
                        frame.append(msg)
                        _, dt = msg
                        t += dt
                    else:
                        break

                    if t >= polling_period:
                        len_correction = 1
                        dt_correction = polling_period - (t - frame[-1][1])
                        break
            else:
                dt_correction += polling_period

            frames, remainder = preprocess(frame[:len(frame) - len_correction], remainder)

            for f in frames:
                output_connection.send(f)

            polling_period = update_polling_period(polling_period)

            frame = frame[len(frame) - len_correction:]

    except KeyboardInterrupt:
        print('Closing.')
    finally:
        print("Exit.")
        midi_in.close_port()
        del midi_in
        output_connection.close()
        if polling_period_connection:
            polling_period_connection.close()
