# -*- coding: utf-8 -*-
"""
Decode the performance from the accompaniment
"""
import numpy as np
import threading
import time

try:
    from score import Note, Chord, Score
except ModuleNotFoundError:
    from .score import Note, Chord, Score


class Accompanist(object):

    def __init__(self, accompaniment_score, performance_codec):

        self.acc_score = accompaniment_score

        self.pc = performance_codec

    def accompaniment_step(self, solo_s_onset, solo_p_onset,
                           velocity, beat_period,
                           articulation):

        # Get next accompanimnet onsets and their
        # respective score iois with respect to the current
        # solo score onset
        next_acc_onsets, next_iois = self.acc_score.next_onsets[solo_s_onset]

        prev_eq_onset = solo_p_onset

        if next_acc_onsets is not None:
            # print('accompaniment step from {0}'.format(next_acc_onsets[0].onset))
            for so, ioi in zip(next_acc_onsets, next_iois):
                # print(ioi, so.onset)

                (perf_onset, so.p_duration,
                 so.velocity, prev_eq_onset) = self.pc.decode_step(
                     ioi=ioi,
                     dur=so.duration,
                     vt=self.acc_score.velocity_trend[so],
                     vd=self.acc_score.velocity_dev[so],
                     lbpr=self.acc_score.log_bpr[so],
                     tim=self.acc_score.timing[so],
                     lart=self.acc_score.log_articulation[so],
                     bp_ave=beat_period,
                     vel_a=velocity,
                     # art_a=articulation,
                     prev_eq_onset=prev_eq_onset)

                so.p_onset = perf_onset

                # if ioi != 0:
                #     so.p_onset = perf_onset

                # else:
                #     if so.onset % 2 == 0:
                #         so.p_onset = perf_onset


class OnlinePerformanceCodec(object):
    def __init__(self, beat_period_ave=0.5,
                 velocity_ave=45,
                 vel_min=20, vel_max=90,
                 init_eq_onset=0.0,
                 mechanical_delay=0.0):

        self.velocity_ave = float(velocity_ave)
        self.bp_ave = float(beat_period_ave)
        self.vel_min = vel_min
        self.vel_max = vel_max
        self.mechanical_delay = mechanical_delay

        self.prev_eq_onset = init_eq_onset

    def decode_step(self, ioi, dur, vt, vd, lbpr,
                    tim, lart, bp_ave, vel_a,
                    # art_a,
                    prev_eq_onset):

        # self.bp_ave = self.moving_average(bp_ave, self.bp_ave)
        self.bp_ave = bp_ave
        self.velocity_ave = vel_a
        # Compute equivalent onsets
        eq_onset = prev_eq_onset + ((2 ** lbpr) * self.bp_ave) * ioi

        # Compute performed onset
        perf_onset = eq_onset - tim - self.mechanical_delay

        # Compute performed duration for each note
        perf_duration = self.decode_duration(dur=dur,
                                             lart=lart,
                                             lbpr=lbpr,
                                             # art_a=art_a,
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

    def decode_duration(self, dur, lart, lbpr,  # art_a,
                        bp_ave):
        perf_duration = ((2 ** lart) * ((2 ** lbpr) * bp_ave) * dur)
        return perf_duration

    def moving_average(self, param_new, param_old, alpha=0.5):

        return alpha * param_old + (1 - alpha) * param_new


if __name__ == '__main__':

    import partitura

    fn = '/Users/aae/Repos/hierarchical_tempo_analysis/datasets/vienna_4x22_corrected/Chopin_op10_no3_p03.match'
    mf = partitura.matchfile.match_to_notearray(fn, expand_grace_notes='d')

    notes = [Note(n['pitch'], n['onset'], n['duration'], n['p_onset'], n['p_duration'], n['velocity'] * .9) for n in mf[:100]]
    acc_score = AccompanimentScore(notes)
