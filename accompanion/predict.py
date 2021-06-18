# -*- coding: utf-8 -*-
"""
Main script for running the ACCompanion

TODO:
* Add visualization functionality
* Document methods
* Simplify and optimize functions
* Cythonize?
"""

import logging
import multiprocessing as mp
import numpy as np
import time
import queue
import json

from collections import OrderedDict
from madmom.utils.midi import MIDIFile

from midi_handling import midi_functions as midif
from score_following.score_follower import score_follower as sf
from score_following.monophonic.dev_tempo_models import LinearSyncModel, LocalIOIModel
from score_following.monophonic.dev_running_average import (OnlineRunningAverage,
                                                            OnlineLocalUpdate)

from accompanist.helper import (load_scores,
                                score_to_sf_score)
from accompanist.expression_tools import (friberg_sundberg_rit,
                                          melody_lead)

# from collections import namedtuple
# import pdb

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(relativeCreated)6d %(processName)s [%(process)d] %(message)s')
LOGGER = logging.getLogger(__name__)


# A class to store the namedtuple to be sent to the queue:
#Output = namedtuple('Output', 'state_idx score_onset velocity beat_period')

def dummy_performer(fn, outport_num):
    """
    Read a MIDI file and play it back as if it were
    an input performance

    Parameters
    ----------
    fn : str
        Filename for the MIDI file.
    out_port : int
        Output port

    Returns
    -------
    performer : mp.Process instance
        Dummy performer
    """

    # Load performance
    performance = MIDIFile.from_file(fn).notes(unit='s')

    # Initialize queue
    q = mp.Queue()
    q_o = mp.Queue()

    # Select inputs as required by produceMidi
    # TODO: Change the order to match the ScoreFollower
    expected_order = np.array([1, 3, 0, 2])

    # NoteID Velocity Onset (s) Duration (s) Onsets (b/fake) clean_buffer
    midi_file = np.column_stack(
        [performance[:, expected_order],
         np.arange(len(performance)),
         np.zeros(len(performance))])

    # Initialize process to simulate a dummy input performer
    performer = mp.Process(name="Lead producer", target=midif.produceMidi,
                           args=(outport_num, q, q_o, q_filter))
    LOGGER.info('Using dummy performer with {0} as input'.format(fn))
    performer.start()

    # initial time
    # TODO: Use pypm.Time() instead
    init_time = time.time()
    # Peformance starts at 0
    # (Is this necessary?)
    midi_file[:, 2] -= midi_file[:, 2].min()
    midi_file[:, 2] += init_time + 2

    for i, mnn in enumerate(midi_file):
        # Send midi messages to produceMidi
        LOGGER.debug('sending MIDI message')
        q.put(mnn)

    return performer


def accompaniment_step(acc_score_dict,
                       solo_score_dict,
                       o, start_idx,
                       cb, velocity, beat_period, articulation,
                       controller_p,
                       controller_target,
                       q_out,
                       mlo=True):
    """
    Compute the performance parameters of the accompaniment
    corresponding to a specific onset in the solo score.

    Parameters
    ----------
    acc_score_dict : dict
        Dictionary containing score information relating to the
        accompaniment score.
    solo_score_dict : dict
        Dictionary containing score information relating to the
        solo score and the corresponding positions in the
        accompaniment score.
    o : float
        Current onset in the solo score, as estimated by the
        score follower.
    start_idx : int (to be deprecated)
        Starting index (to be deprecated. For now must always be
        zero)
    cb : bool (or 1 or 0)
        clean the buffer. See `midi_functions.produceMidi`
        for more details.
    velocity : float
        Current estimate of the MIDI velocity of the solo.
    beat_period : float
        Current estimate of the beat period.
    articulation : float
        Current estimate of the articulation ratio.
    controller_p : float or None
        Value of the dial of the Griffin Power Mate
    controller_target : str
        Target to be controlled by the Griffin Power Mate
        Valid targets are 'dynamics', 'dynamics_trend',
        'timing', 'timing_trend', 'articulation'.
    q_out : Queue
        Queue to send output information to the
        MIDI producer. See `midi_functions.produceMidi`

    Returns
    -------
    cb : 0
        Do not clean the buffer (see above).
    """
    # print "Beginning of Acc Step..."
    # Initialize parameters for the Power Mate
    c_p = 3 ** controller_p
    dyn_c = 1.0
    a_dyn_tc = 1.0
    b_dyn_tc = 0.0
    tim_c = 1.0
    tim_tc = 1.0
    art_c = 1.0
    vel_t = 0.50 * velocity

    if controller_target == 'dynamics':
        dyn_c = 3 ** controller_p
    elif controller_target == 'dynamics_trend':
        a_dyn_tc = (controller_p + 1.0) / 2.
        b_dyn_tc = 1 - a_dyn_tc
        vel_t = velocity
    elif controller_target == 'timing':
        tim_c = 3 * controller_p
    elif controller_target == 'timing_trend':
        tim_tc = 1.5 ** controller_p
    elif controller_target == 'articulation':
        art_c = c_p

    # print "Acc_Step: After Controller... "

    # Ceus MIDI delay insertion:
    next_onsets = solo_score_dict[o][7][start_idx:]
    init_time = time.time()
    a_onsets = ((solo_score_dict[o][9] *
                 beat_period * tim_tc) +
                init_time)[start_idx:]

    # print "Acc_step: After onset stuff..."
    # TODO:
    # * handle already performed onsets

    for i, (ao_s, ao_p) in enumerate(zip(next_onsets, a_onsets)):
        a_pitches = acc_score_dict[ao_s][0]

        a_durations = (acc_score_dict[ao_s][1] *
                       (acc_score_dict[ao_s][6] * art_c) *
                       articulation)
        # Restrict durations larger to the range of 0.075
        # and 3 seconds
        np.clip(a_durations, a_min=0.075,
                a_max=3.0, out=a_durations)

        a_velocities = np.round(
            (acc_score_dict[ao_s][2] *
             (a_dyn_tc * vel_t + b_dyn_tc * 64) +
             (acc_score_dict[ao_s][3] * dyn_c)))
        # Restrict velocities to 30, 120
        np.clip(a_velocities, a_min=10,
                a_max=120, out=a_velocities)
        if i > 0:
            a_timings = (acc_score_dict[ao_s][5] * tim_c) + ao_p
        else:
            a_timings = np.zeros_like(acc_score_dict[ao_s][5]) + ao_p

        if mlo:
            ml = melody_lead(pitch=a_pitches, velocity=velocity,
                             lead=0.02)
            a_timings += ml

        for a_pitch, a_duration, a_vel, a_timing in zip(
                a_pitches, a_durations, a_velocities, a_timings):

            msg = np.array([a_pitch, a_vel,
                            a_timing, a_duration,
                            ao_s, cb])
            cb = 0
            q_out.put(msg)

    return cb


class ProcessTerminator(object):
    """
    A class that'll be back to terminate
    running processes.
    """

    def __init__(self, *args):
        self.processes_to_terminate = []
        if args is not None:
            self.processes_to_terminate = [p for p in args]

    def add(self, p):
        self.processes_to_terminate.append(p)

    @property
    def terminate(self):

        for p in self.processes_to_terminate:
            p.join()
            p.terminate()


def main(acc_fn, solo_fn, bpm,
         inport_num, outport_num,
         bm_models=None,
         q_gui=None,
         q_filter=None,
         use_dummy_performer=False,
         perf_midi_fn=None,
         use_powermate=None,
         standardize=None):
    """
    Main ACCompanator script.

    Parameters
    ----------
    acc_fn : str
        Path to the accompaniment score in MusicXML format.
    solo_fn : str
        Path to the solo score in MusicXML format.
    bpm : float
        Initial beats per minute.
    inport_num : int
        Index of the MIDI input port
    outport_num : int
        Index of the MIDI output port
    bm_models : str
        Path to the precomputed outputs of the Basis-Mixer
    use_dummy_performer : bool
        Play a MIDI file as the solo part.
    perf_midi_fn : str or None
        Path to the MIDI file to be used as a solo part.
    use_powermate : str or None
        Name of the target to be controlled by the
        Griffin Power Mate.
    """
    config_filepath = './score_following/config/follower_config.json'
    # Initialize queue for using PowerMate
    q_dial = mp.Queue()
    if use_powermate is not None:
        import accompanist.powermate as powermate
        controller = mp.Process(
            target=powermate.run_device, args=(q_dial,))
        controller_target = use_powermate

    else:
        controller = None
        controller_target = None

    # Initialize beat period
    beat_period = 60.0 / bpm

    # Load scores (all of them):
    acc_score, solo_score, acc_xml_part = load_scores(acc_fn, solo_fn)

    # unique onsets in the accompaniment
    unique_acc_onsets = np.unique(acc_score['onset'])
    unique_acc_onsets = np.column_stack(
        [unique_acc_onsets, np.zeros_like(unique_acc_onsets)])
    # Unique onsets in the solo
    unique_solo_onsets = np.unique(solo_score['onset'])

    # q_offset = mp.Queue()
    q_input = mp.Queue()

    # Extract the configuration for the score follower:
    # Get the config file:
    with open(config_filepath, "r") as read_file:
        config = json.load(read_file)

    # Create the score follower here:
    score_follower = sf.ScoreFollower(score_type='xml', score=solo_score,
                                      config=config, q_gui=q_gui)

    bm_models = None
    # Load precomputed models from the basis mixer
    if bm_models is not None and acc_xml_part is not None:

        LOGGER.info('Using pre-computed basis-mixer targets')
        exp_targets = dict(
            list(np.load(bm_models[0]).items()))
        vt = exp_targets['vt'].astype(np.float)
        vd = exp_targets['vd'].astype(np.float)
        ib = exp_targets['ib'].astype(np.float)
        ti = exp_targets['ti'].astype(np.float)
        ar = exp_targets['ar'].astype(np.float)

        vt_norm = vt / vt.mean()
        ibi_norm = 2 ** (ib - ib.mean())
        ar = ar / ar.mean() * 1.45

        if standardize is not None:
            print('Standardize {0}'.format(standardize))
        if standardize == 'vt_norm':
            vt_norm = (vt_norm - vt_norm.mean()) / (vt_norm.std())
        elif standardize == 'vd':
            vd = (vd - vd.mean()) / (vd.std())
        elif standardize == 'ib_norm':
            ibi_norm = (ibi_norm - ibi_norm.mean()) / (ibi_norm.std())
        elif standardize == 'ti':
            ti = (ti - ti.mean()) / (ti.std())
        elif standardize == 'ar':
            ar = (ar - ar.mean()) / (ar.std())

    else:
        LOGGER.info('Using deadpan version...')
        vt_norm = np.ones(len(unique_acc_onsets))
        ibi_norm = np.ones(len(unique_acc_onsets))
        vd = np.zeros(len(acc_score))
        ti = np.zeros(len(acc_score))
        ar = np.ones(len(acc_score))

    final_rit = True
    if final_rit:
        final_rit_idx = np.where(
            unique_acc_onsets[:, 0] >= solo_score['onset'].max())[0]

        final_art_idx = np.where(
            acc_score['onset'] >= solo_score['onset'].max())[0]
        ritard_curve = friberg_sundberg_rit(
            len_c=len(final_rit_idx), r_w=0.5, r_q=1.1)
        ibi_norm[final_rit_idx] *= ritard_curve
        ar[final_art_idx] *= 1.5

    # Precompute joint score onsets
    acc_iois = np.diff(unique_acc_onsets[:, 0])
    num_acc_onsets = len(unique_acc_onsets)
    acc_score_dict = OrderedDict()

    for i, on in enumerate(unique_acc_onsets[:, 0]):

        idxs = np.where(acc_score['onset'] == on)[0]
        next_onsets = unique_acc_onsets[i:, 0]
        next_iois_b = np.r_[0, acc_iois[slice(i, num_acc_onsets)]]
        # dimensionless next onsets (to be multiplied by ave beat period
        next_onsets_d = np.cumsum(next_iois_b * ibi_norm[i:])
        # 0:Pitch 1:Duration 2:vel_trend (norm)
        # 3:vel_dev 4:ibi (norm) 5:timing 6:articulation
        # 7:next_onsets(b) 8:next_iois(b) 9:next_onsets (d)
        # 10:n_notes 11:index in unique_acc_onsets
        # TODO: Compute performed onsets here? This would
        # avoid having to iterate over the dictionary
        acc_score_dict[on] = (acc_score['pitch'][idxs],
                              acc_score['duration'][idxs],
                              vt_norm[i],
                              vd[idxs],
                              ibi_norm[i],
                              ti[idxs],
                              ar[idxs],
                              next_onsets,
                              next_iois_b,
                              next_onsets_d,
                              len(idxs), i)

    solo_score_dict = dict()
    for i, on in enumerate(unique_solo_onsets):
        acc_idx = np.min(np.where(unique_acc_onsets[:, 0] >= on)[0])
        acc_idxs = np.where(
            acc_score['onset'] == unique_acc_onsets[acc_idx, 0])
        next_acc_onsets = unique_acc_onsets[acc_idx:, 0]
        ioi_init = next_acc_onsets[0].min() - on
        next_iois = np.r_[ioi_init, np.diff(next_acc_onsets)]
        next_onsets = np.cumsum(next_iois)
        next_onsets_d = np.cumsum(next_iois * ibi_norm[acc_idx:])
        # 0:Pitch 1:Duration 2:vel_trend (norm)
        # 3:vel_dev 4:ibi (norm) 5:timing 6:articulation
        # 7:next_onsets(b) 8:next_iois(b) 9:next_onsets (d)
        # 10:n_notes 11:index in unique_acc_onsets
        # 12: index in unique_solo_onsets
        solo_score_dict[on] = (acc_score['pitch'][acc_idxs],
                               acc_score['duration'][acc_idxs],
                               vt_norm[acc_idx],
                               vd[acc_idxs],
                               ibi_norm[acc_idx],
                               ti[acc_idxs],
                               ar[acc_idxs],
                               next_acc_onsets,
                               next_iois,
                               next_onsets_d,
                               len(acc_idxs),
                               acc_idx,
                               i)

    input_process = mp.Process(
        name="Follower",
        target=score_follower.runWithMidiInput,
        args=(inport_num, q_input, ))
    print("Starting follower:")
    input_process.start()

    if use_dummy_performer:
        # Load performance
        input_performer = dummy_performer(perf_midi_fn, inport_num)
        t = ProcessTerminator(input_performer, input_process)

    else:
        t = ProcessTerminator(input_process)

    # Queues for doing more awesome stuff
    # Probably these need to be deleted
    q_out = mp.Queue()
    q_acc = mp.Queue()

    accompanist = mp.Process(
        name="Accompaniment Producer",
        target=midif.produceMidi, kwargs=dict(
            outport_num=outport_num,
            q_in=q_out,
            q_out=q_acc,
            q_filter=q_filter,
            q_gui=q_gui))
    LOGGER.info('Starting accompanist')
    accompanist.start()

    velocity_model = 'OnlineLocalUpdate'

    if velocity_model == 'OnlineRunningAverage':
        vm = OnlineRunningAverage(order=len(solo_score) // 20)
    if velocity_model == 'OnlineLocalUpdate':
        vm = OnlineLocalUpdate(a_min=30, a_max=120)

    articulation_model = 'OnlineRunningAverage'
    if articulation_model == 'OnlineRunningAverage':
        am = OnlineRunningAverage(order=5)
    elif articulation_model == 'OnlineLocalUpdate':
        am = OnlineLocalUpdate(a_min=0.25, a_max=1.5)

    velocity = 80
    articulation = 1.0

    timing_model = 'LocalIOIModel'

    if timing_model == 'LinearSyncModel':
        tm = LinearSyncModel(
            init_beat_period=beat_period,
            init_score_onset=unique_acc_onsets[0, 0],
            eta_t=0.1, eta_p=0.3)
    elif timing_model == 'LocalIOIModel':
        tm = LocalIOIModel(
            init_beat_period=beat_period,
            init_score_onset=unique_acc_onsets[0, 0])

    off_info = None
    est_info = None
    p_update = None
    controller_p = 0.0
    if controller is not None:
        t.add(controller)
        controller.start()
        controller_init = 0.0
        p_update = None
        mult = 0.1

    while True:
        try:
            # print "Another repeat of loop..."
            # [0:onset (s), 1:position index, 2:tempo, 3:onset (beats),
            #  4:pitch, 5:behaviour, 6:velocity, 7:offset (beats)]
            LOGGER.debug("waiting for input")
            try:
                est_info = q_input.get_nowait()
                # print "Getting stuf..."
            except queue.Empty:
                est_info = None
                pass

            # try:
            #     # Get info from offsets
            #     off_info = q_offset.get_nowait()
            # except Queue.Empty:
            #     off_info = None
            #     pass

            if controller is not None:
                try:
                    p_update = q_dial.get_nowait()
                except queue.Empty:
                    p_update = None
                    pass

            if est_info is not None:
                # if est_info[0, 5] == 1:
                    # current solo onset
                    # cso = est_info[0, 3]
                    # current_perf_solo_onset = est_info[0, 0]
                    # velocity = vm.update(est_info[0, 6])
                    # beat_period = tm.update_beat_period(
                    #     current_perf_solo_onset, cso)

                beat_period = est_info[3]
                velocity = est_info[2] + 0.5 * est_info[2]
                onset = est_info[1]
                state_pred = est_info[0]
                # print "After parameter extraction..."

                # Clean buffer variable:
                cb = 1

                # TODO: Handle repated onsets
                # if not np.isnan(cso):
                # If it is not an inserted state:
                # print "Est info 0: ",est_info[0]

                if state_pred % 2 == 0:
                    # print "Just before accompaniment step..."
                    # pdb.set_trace()
                    cb = accompaniment_step(
                        acc_score_dict=acc_score_dict,
                        solo_score_dict=solo_score_dict,
                        o=onset,
                        start_idx=0,
                        cb=cb, velocity=velocity,
                        beat_period=beat_period,
                        articulation=articulation,
                        controller_p=controller_p,
                        controller_target=controller_target,
                        q_out=q_out)

                # print "Getting stuff after..."

            # # Get info from articulation
            # if off_info is not None:
            #     p_duration = off_info[0, 0]
            #     s_duration = off_info[0, 7]
            #     # Update articulation
            #     articulation = am.update(
            #         np.nan_to_num(p_duration / (s_duration + 1e-10)))

            if p_update is not None:
                left, right, button = p_update

                controller_p += mult * (right - left)

                controller_p = np.clip(controller_p,
                                       a_min=-1, a_max=1)

                if button:
                    controller_p = controller_init

            time.sleep(1e-5)
            # print "After sleep..."
        except KeyboardInterrupt:
            break

    t.terminate


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('ACCompanion v 0.01')

    parser.add_argument('piece',
                        help='Name of the piece')
    parser.add_argument('--use-dummy-performer',
                        help='Use a MIDI file as the performer',
                        action='store_true', default=False)
    parser.add_argument('--bpm',
                        help='Beats per minute',
                        type=float,
                        default=120.0)
    parser.add_argument('--bm-models', nargs='+',
                        help="One or more model files")
    parser.add_argument('--use-powermate',
                        help='Use Griffin PowerMate to modify a target',
                        default=None)
    args = parser.parse_args()

    use_xml = True

    if use_xml:
        acc_fn = 'midi_data/{0}/acc-score.xml'.format(args.piece)
        solo_fn = 'midi_data/{0}/humanlead-score.xml'.format(args.piece)
    else:
        acc_fn = 'midi_data/{0}/acc-score.mid'.format(args.piece)
        solo_fn = 'midi_data/{0}/humanlead-score.mid'.format(args.piece)

    perf_midi_fn = 'midi_data/{0}/humanlead.mid'.format(args.piece)

    # List of port numbers
    q_display_ports = mp.Queue()

    # Display the available Midi ports (in a separate process):
    print_process = mp.Process(target=midif.printPorts,
                               args=(q_display_ports,))
    print_process.start()

    # List of port numbers
    ports = q_display_ports.get()

    # Wait for the process to terminate:
    print_process.join()
    print_process.terminate()

    # Select input and output ports
    inport_num = midif.set_port_number(ports[0], 'input')
    outport_num = midif.set_port_number(ports[1], 'output')

    accompanist = mp.Process(name='ACCompanion',
                             target=main,
                             kwargs=dict(
                                 acc_fn=acc_fn,
                                 solo_fn=solo_fn,
                                 perf_midi_fn=perf_midi_fn,
                                 bpm=args.bpm,
                                 inport_num=inport_num,
                                 outport_num=outport_num,
                                 bm_models=args.bm_models,
                                 use_dummy_performer=args.use_dummy_performer,
                                 use_powermate=args.use_powermate))
    try:
        accompanist.start()
    except KeyboardInterrupt:
        accompanist.terminate()
