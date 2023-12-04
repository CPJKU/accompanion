# -*- coding: utf-8 -*-
"""
Decode the performance from the accompaniment
"""
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from partitura.score import Part

from accompanion.accompanist.score import AccompanimentScore
from accompanion.accompanist.tempo_models import SyncModel
from accompanion.config import CONFIG
from accompanion.score_follower.note_tracker import NoteTracker
from accompanion.utils.expression_tools import friberg_sundberg_rit


class OnlinePerformanceCodec(object):
    """
    An Online Version of the Performance Codec.

    Parameters
    ----------
    part: partitura.score.Part (optional)
        The solo part.
    note_tracker : NoteTracker (optional)
        The note tracker.
    beat_period_ave : float (optional)
        The average beat period.
    velocity_ave : int (optional)
        The average velocity. Is used when the moving average cannot be computed.
    vel_min : int (optional)
        The minimum velocity accompaniment.
    vel_max : int (optional)
        The maximum velocity of the accompaniment.
    velocity_ma_alpha : float (optional)
        The alpha parameter for the moving average of the velocity.
    init_eq_onset : float (optional)
        The initial equalized onset.
    mechanical_delay : float (optional)
        The mechanical delay of the accompaniment (to be used with a mechanical piano).
        This refers to the delay of the message arriving to the piano and the 
        mechanical piano actually producing the sound.
    tempo_model : SyncModel (optional)
        The tempo model to be used from available models in accompanist 
        tempo_models.py
    vel_prev : int (optional)
        The previous velocity.
    articulation_prev : float (optional)
        The previous articulation.
    articulation_ma_alpha: float (optional)
        The alpha parameter for the moving average of the articulation.
    """

    def __init__(
        self,
        part: Optional[Part] = None,
        note_tracker: Optional[NoteTracker] = None,
        beat_period_ave: float = 0.5,
        velocity_ave: Union[int, float] = 45,
        vel_min: int = 20,
        vel_max: int = 90,
        velocity_ma_alpha: float = 0.6,
        init_eq_onset: float = 0.0,
        mechanical_delay: float = 0.0,
        tempo_model: Optional[SyncModel] = None,
        vel_prev: float = 60,
        articulation_prev: float = 1.0,
        articulation_ma_alpha: float = 0.4,
        **kwargs
    ) -> None:
        self.velocity_ave: float = float(velocity_ave)
        self.bp_ave: float = float(beat_period_ave)
        self.vel_min: int = vel_min
        self.vel_max: int = vel_max
        self.velocity_ma_alpha: float = velocity_ma_alpha
        self.articulation_ma_alpha: float = articulation_ma_alpha
        self.mechanical_delay: float = mechanical_delay
        self.prev_eq_onset: float = init_eq_onset
        self.part: Optional[Part] = part
        self.note_tracker: Optional[NoteTracker] = note_tracker
        self.tempo_model: Optional[SyncModel] = tempo_model
        self.vel_prev: float = vel_prev
        self.articulation_prev: float = articulation_prev
        self.kwargs: dict = kwargs

    def encode_step(self) -> Tuple[float, float]:
        """
        Encode the performance of the soloist into expressive
        parameters
        """
        try:
            self.vel_prev = moving_average_online(
                np.max(self.note_tracker.velocities[-1]),
                self.vel_prev,
                self.velocity_ma_alpha,
            )
        except Exception:
            self.vel_prev = self.velocity_ave

        articulation_list = []

        for id in self.note_tracker.recently_closed_snotes:
            notated_duration = self.note_tracker.note_dict[id][1]
            perf_duration = self.note_tracker.note_dict[id][3]

            articulation_list.append(perf_duration / (notated_duration * self.bp_ave))

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
                    articulation, self.articulation_prev, self.articulation_ma_alpha
                ),
                a_min=0.3,
                a_max=1.5,
            )

        vel_scale = self.kwargs.get("velocity_solo_scale", 0.8)

        return vel_scale * self.vel_prev, self.articulation_prev

    def decode_step(
        self,
        ioi: float,
        dur: Union[np.ndarray, float],
        vt: Union[np.ndarray, float],
        vd: Union[np.ndarray, float],
        lbpr: Union[np.ndarray, float],
        tim: Union[np.ndarray, float],
        lart: Union[np.ndarray, float],
        bp_ave: float,
        vel_a: float,
        art_a: float,
        prev_eq_onset: Optional[float] = None,
    ) -> Tuple[
        Union[np.ndarray, float],
        Union[np.ndarray, float],
        Union[np.ndarray, int],
        float,
    ]:
        """
        Decode the accompaniment part (in terms of performed onsets, durations)
        and MIDI velocity

        Parameters
        ----------
        ioi: float
            Score inter-onset interval for the next onset (in beats)
        dur: Union[np.ndarray, float]
            Duration of the notes (in beats)
        vt: Union[np.ndarray, float]
            Trend in MIDI velocity
        vd: Union[np.ndarray, float]
            Deviations in MIDI velocity from the Trend
        lbpr: Union[np.ndarray, float]
            Logarithm of the beat period ratio (ratio of local beat period to average
            beat period)
        tim: Union[np.ndarray, float]
            Micro-timing deviations for the notes
        lart: Union[np.ndarray, float]
            Logarithm of the articulation ratio
        bp_ave: float
            Average beat period
        vel_a: float
            Average MIDI velocity
        art_a: float
            Average articulation ratio
        prev_eq_onset: Optional[float]
            Previous equivalent onset time (in seconds).

        Returns
        -------
        perf_onset: np.ndarray or float
            Performed onset time of the notes in seconds
        perf_duration: np.ndarray or float
            Performed duration of the notes in seconds
        perf_vel: np.ndarray or int
            Performed MIDI velocities of the notes
        eq_onset: float
            Equivalent onset time (the average value of the onset of all notes
            in a chord).
        """
        self.bp_ave = bp_ave
        self.velocity_ave = vel_a
        # Compute equivalent onsets
        eq_onset = prev_eq_onset + self.bp_ave * ioi

        # Compute performed onset
        perf_onset = eq_onset - tim - self.mechanical_delay

        # Compute performed duration for each note
        perf_duration = self.decode_duration(
            dur=dur,
            lart=lart,
            lbpr=lbpr,
            art_a=art_a,
            bp_ave=self.bp_ave,
        )

        # Compute velocity for each note
        perf_vel = self.decode_velocity(vt, vd, self.velocity_ave)

        return perf_onset, perf_duration, perf_vel, eq_onset

    def decode_velocity(
        self,
        vt: Union[np.ndarray, float],
        vd: Union[np.ndarray, float],
        vel_ave: float,
    ) -> Union[np.ndarray, int]:
        """
        Decode MIDI velocity

        Parameters
        ----------
        vt: Union[np.ndarray, float]
            Trend in MIDI velocity
        vd: Union[np.ndarray, float]
            Deviations in MIDI velocity from the Trend
        vel_a: float
            Average MIDI velocity

        Return
        ------
        perf_vel: np.ndarray or int
            Performed MIDI velocities of the notes
        """
        # Add options for normalization
        perf_vel = np.clip(vt * vel_ave - vd, self.vel_min, self.vel_max).astype(int)
        return perf_vel

    def decode_duration(
        self,
        dur: Union[np.ndarray, float],
        lart: Union[np.ndarray, float],
        lbpr: Union[np.ndarray, float],
        art_a: float,
        bp_ave: float,
    ) -> Union[np.ndarray, float]:
        """
        Decode performed duration

        Parameters
        ----------
        dur: Union[np.ndarray, float]
            Duration of the notes (in beats)
        lart: Union[np.ndarray, float]
            Logarithm of the articulation ratio
        lbpr: Union[np.ndarray, float]
            Logarithm of the beat period ratio (ratio of local beat period to average
            beat period)
        art_a: float
            Average articulation ratio
        bp_ave: float
            Average beat period

        Returns
        -------
        perf_duration: np.ndarray or float
            Performed duration of the notes in seconds
        """
        # TODO: check expected articulation in solo (legato, staccato),
        # and only use it if it is the "same" as notated in the
        # accompaniment score

        art_a = np.clip(art_a, 0, 1)

        perf_duration = (2**lart) * ((2**lbpr) * bp_ave) * dur * art_a
        np.clip(perf_duration, a_min=0.01, a_max=4, out=perf_duration)
        return perf_duration


class Accompanist(object):
    """
    The Accompanist class is responsible for decoding the performance
    from the accompaniment.

    Parameters
    ----------
    accompaniment_score : AccompanimentScore
        The accompaniment score.
    performance_codec : PerformanceCodec
        The performance codec.
    """

    acc_score: AccompanimentScore
    pc: OnlinePerformanceCodec

    def __init__(
        self,
        accompaniment_score: AccompanimentScore,
        performance_codec: OnlinePerformanceCodec,
        decoder_kwargs: Optional[Dict] = None,
    ) -> None:
        self.acc_score = accompaniment_score

        self.pc = performance_codec

        self.decoder_kwargs = decoder_kwargs if decoder_kwargs is not None else {}

        self.prev_eq_onsets = np.zeros(
            len(self.acc_score.ssc.unique_onsets), dtype=float
        )
        self.step_counter = 0
        self.bp_prev = None
        self.rit_curve = friberg_sundberg_rit(
            len_c=self.decoder_kwargs.get("rit_len", CONFIG["RIT_LEN"]),
            r_w=self.decoder_kwargs.get("rit_w", CONFIG["RIT_W"]),
            r_q=self.decoder_kwargs.get("rit_q", CONFIG["RIT_Q"]),
        )
        self.rit_counter = 0

        self.tempo_change_curve = dict()

        j = 0
        all_chords = self.acc_score.solo_score_dict[
            list(self.acc_score.solo_score_dict.keys())[0]
        ][0]
        self.num_chords = len(all_chords)
        for i, so in enumerate(all_chords):
            self.tempo_change_curve[so] = 1.0
            if self.num_chords - i <= self.decoder_kwargs.get(
                "rit_len", CONFIG["RIT_LEN"]
            ):
                self.tempo_change_curve[so] = self.rit_curve[j]
                j += 1

    def accompaniment_step(
        self,
        solo_s_onset: float,
        solo_p_onset: float,
        tempo_expectations: Optional[Callable[[float], float]] = None,
    ) -> None:
        """
        Update the performance of the accompaniment part given the latest
        information from the solo performance. This method does not
        return the parameters of the performance, but updates the
        notes in the accompaniment score sequencer directly.

        Parameters
        ----------
        solo_s_onset : float
            Currently performed score onset (in beats)
        solo_p_onset : float
            Current performed score onset time (in seconds)
        tempo_expectations: Callable
            A callable method that outputs the tempo expectations
            given the corresponding score onset position.
        """
        # Get next accompaniment onsets and their
        # respective score iois with respect to the current
        # solo score onset

        next_acc_onsets, next_iois, _, _, suix = self.acc_score.solo_score_dict[
            solo_s_onset
        ]

        # if self.step_counter > 0:
        #     beat_period, eq_onset = self.pc.tempo_model(solo_p_onset,
        #                                                 solo_s_onset)
        # else:
        #     beat_period = self.pc.bp_ave
        #     eq_onset = solo_p_onset

        beat_period, eq_onset = self.pc.tempo_model(solo_p_onset, solo_s_onset)

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

                (
                    perf_onset,
                    so.p_duration,
                    so.velocity,
                    prev_eq_onset,
                ) = self.pc.decode_step(
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
                    prev_eq_onset=prev_eq_onset,
                )

                # if i == 0:
                #     print(so.onset, perf_onset, ioi, bp_ave, solo_p_onset)

                if ioi != 0 or self.step_counter == 0:
                    so.p_onset = perf_onset
                    # if i == 0:
                    #     print(
                    #         "accompaniment step",
                    #         so.onset,
                    #         ioi,
                    #         so.p_onset,
                    #         perf_onset,
                    #         perf_onset - solo_p_onset,
                    #     )

        self.step_counter += 1


def moving_average_online(
    param_new: Union[np.ndarray, float],
    param_old: Union[np.ndarray, float],
    alpha: float = 0.5,
) -> Union[np.ndarray, float]:
    """
    Step of the online computation of the moving average (MA) value of
    a time series

    Parameters
    ----------
    param_new: Union[np.ndarray, float]
        New observation
    param_old: Union[np.ndarray, float]
        Previous estimate of the moving average
    alpha: float
        Smoothing factor (must be between 0 and 1).
        A value closer to 1 changes the MA value very slowly, while a
        value closer to 0 "forgets" the previous estimate and
        takes always the most recent value.

    Returns
    -------
    ma: Union[np.ndarray, float]
        New estimate of the moving average.
    """
    ma = alpha * param_old + (1 - alpha) * param_new
    return ma


def moving_average_offline(
    parameter: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Smooth a curve by taking its moving average.

    Parameters
    ----------
    parameter: np.ndarray
        The input 1D time series.
    alpha: float
        Smoothing factor (must be between 0 and 1).
        A value closer to 1 changes the MA value very slowly, while a
        value closer to 0 "forgets" the previous estimate and
        takes always the most recent value.

    Returns
    -------
    ma: Union[np.ndarray, float]
        The smoothed curve.
    """
    ma = np.zeros_like(parameter)
    ma[0] = parameter[0]

    for i in range(1, len(parameter)):
        ma[i] = alpha * ma[i - 1] + (1 - alpha) * parameter[i]

    return ma


if __name__ == "__main__":
    pass
