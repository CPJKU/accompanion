# -*- coding: utf-8 -*-
"""
Decode the performance from the accompaniment
"""
import numpy as np
from accompanion.accompanist.expression_tools import friberg_sundberg_rit


RIT_LEN = 24

class Accompanist(object):

    def __init__(self, accompaniment_score, performance_codec):

        self.acc_score = accompaniment_score

        self.pc = performance_codec
        self.prev_eq_onsets = np.zeros(
            len(self.acc_score.ssc.unique_onsets),
            dtype=float)
        self.step_counter = 0
        self.bp_prev = None
        self.rit_curve = friberg_sundberg_rit(RIT_LEN, r_w=0.75)
        self.rit_counter = 0
        # self.num_acc_onsets = len(self.acc_score.ssc.unique_onsets)

        self.tempo_change_curve = dict()

        j = 0
        all_chords = self.acc_score.solo_score_dict[list(self.acc_score.solo_score_dict.keys())[0]][0]
        self.num_chords = len(all_chords)
        for i, so in enumerate(all_chords):
            self.tempo_change_curve[so] = 1.
            if self.num_chords - i <= RIT_LEN:
                self.tempo_change_curve[so] = self.rit_curve[j]
                j += 1

    def accompaniment_step(self, solo_s_onset, solo_p_onset,
                           tempo_expectations=None):

        # Get next accompaniment onsets and their
        # respective score iois with respect to the current
        # solo score onset

        next_acc_onsets, next_iois,  \
            _, _, suix = self.acc_score.solo_score_dict[solo_s_onset]

        # if self.step_counter > 0:
        #     beat_period, eq_onset = self.pc.tempo_model(solo_p_onset,
        #                                                 solo_s_onset)
        # else:
        #     beat_period = self.pc.bp_ave
        #     eq_onset = solo_p_onset
        
        beat_period, eq_onset = self.pc.tempo_model(solo_p_onset,
                                                    solo_s_onset)

        # print(self.step_counter, beat_period)

        velocity, articulation = self.pc.encode_step()

        # if solo_p_onset is not None:
        self.prev_eq_onsets[suix] = eq_onset
        prev_eq_onset = self.prev_eq_onsets[suix]
        
        if next_acc_onsets is not None:

            for i, (so, ioi) in enumerate(zip(next_acc_onsets, next_iois)):

                if tempo_expectations is not None and i != 0:
                    bp_ave = tempo_expectations(so.onset)
                else:
                    bp_ave = beat_period

                    t_factor = self.tempo_change_curve[so]
                    bp_ave *= t_factor


                (perf_onset, so.p_duration,
                 so.velocity, prev_eq_onset) = self.pc.decode_step(
                     ioi=ioi,
                     dur=so.duration,
                     # TODO: This will need to be changed when
                     # the expressive parameters are computed in
                     # in real time.
                     vt=self.acc_score.velocity_trend[so],
                     vd=self.acc_score.velocity_dev[so],
                     lbpr=self.acc_score.log_bpr[so],
                     tim=self.acc_score.timing[so],
                     lart=self.acc_score.log_articulation[so],
                     bp_ave=bp_ave,
                     vel_a=velocity,
                     art_a=articulation,
                     prev_eq_onset=prev_eq_onset)

                # if i == 0:
                #     print(so.onset, perf_onset, ioi, bp_ave, solo_p_onset)
                
                if (ioi != 0 or self.step_counter == 0):
                    # print('accompaniment step')
                    so.p_onset = perf_onset
                    
                    

        self.step_counter += 1


class OnlinePerformanceCodec(object):
    def __init__(self, part=None,
                 note_tracker=None,
                 beat_period_ave=0.5,
                 velocity_ave=45,
                 vel_min=20, vel_max=90,
                 velocity_ma_alpha=0.6,
                 init_eq_onset=0.0,
                 mechanical_delay=0.0,
                 tempo_model=None,
                 vel_prev=60,
                 articulation_prev=1,
                 articulation_ma_alpha=0.4,
                 **kwargs):

        self.velocity_ave = float(velocity_ave)
        self.bp_ave = float(beat_period_ave)
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.velocity_ma_alpha = velocity_ma_alpha
        self.articulation_ma_alpha = articulation_ma_alpha

        self.mechanical_delay = mechanical_delay

        self.prev_eq_onset = init_eq_onset
        self.part = part
        self.note_tracker = note_tracker
        self.tempo_model = tempo_model

        self.vel_prev = vel_prev
        self.articulation_prev = articulation_prev
        self.kwargs = kwargs

    def encode_step(self):
        try:
            self.vel_prev = moving_average_online(
                np.max(self.note_tracker.velocities[-1]),
                self.vel_prev,
                self.velocity_ma_alpha
            )
        except:
            self.vel_prev = self.velocity_ave

        articulation_list = []

        for id in self.note_tracker.recently_closed_snotes:
            notated_duration = self.note_tracker.note_dict[id][1]
            perf_duration = self.note_tracker.note_dict[id][3]

            articulation_list.append(
                perf_duration/(notated_duration*self.bp_ave)
            )

        # reset recently_closed_snotes
        self.note_tracker.recently_closed_snotes = []

        if len(articulation_list) > 0:
            articulation = np.mean(articulation_list)
        else:
            articulation = 0

        # beta = 0.4
        if articulation > 0:
            # TODO: Find a better parametrization of the articulation
            self.articulation_prev = np.clip(
                moving_average_online(
                    articulation,
                    self.articulation_prev,
                    self.articulation_ma_alpha
                ),
                a_min=0.3,
                a_max=1.5
            )

        vel_scale = self.kwargs.get("velocity_solo_scale", 0.8)
        
        return vel_scale * self.vel_prev, self.articulation_prev

    def decode_step(self, ioi, dur, vt, vd, lbpr,
                    tim, lart, bp_ave, vel_a, art_a,
                    prev_eq_onset=None):

        self.bp_ave = bp_ave
        self.velocity_ave = vel_a
        # Compute equivalent onsets
        eq_onset = prev_eq_onset + self.bp_ave * ioi

        # Compute performed onset
        perf_onset = eq_onset - tim - self.mechanical_delay

        # Compute performed duration for each note
        perf_duration = self.decode_duration(dur=dur,
                                             lart=lart,
                                             lbpr=lbpr,
                                             art_a=art_a,
                                             bp_ave=self.bp_ave)

        # Compute velocity for each note
        perf_vel = self.decode_velocity(vt, vd, self.velocity_ave)

        return perf_onset, perf_duration, perf_vel, eq_onset

    def decode_velocity(self, vt, vd, vel_ave):
        # Add options for normalization
        perf_vel = np.clip(vt * vel_ave - vd,
                           self.vel_min,
                           self.vel_max).astype(np.int)
        return perf_vel

    def decode_duration(self, dur, lart, lbpr, art_a, bp_ave):
        # TODO: check expected articulation in solo (legato, staccato),
        # and only use it if it is the "same" as notated in the
        # accompaniment score

        art_a = np.clip(art_a, 0, 1)

        perf_duration = ((2 ** lart) * ((2 ** lbpr) * bp_ave) * dur * art_a)
        np.clip(perf_duration, a_min=0.01, a_max=4, out=perf_duration)
        return perf_duration


def moving_average_online(param_new, param_old, alpha=0.5):
    return alpha * param_old + (1 - alpha) * param_new


def moving_average_offline(parameter, alpha=0.5):
    ma = np.zeros_like(parameter)
    ma[0] = parameter[0]

    for i in range(1, len(parameter)):
        ma[i] = alpha * ma[i - 1] + (1 - alpha) * parameter[i]

    return ma


if __name__ == '__main__':
    pass
