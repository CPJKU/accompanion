# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import interp1d


class SyncModel(object):
    """
    Base class for synchronization models
    """

    def __init__(self, init_beat_period=0.5, init_score_onset=0):
        self.beat_period = init_beat_period
        self.prev_score_onset = init_score_onset
        self.prev_perf_onset = None
        self.est_onset = None
        self.asynchrony = 0.0
        self.has_tempo_expectations = False
        # Count how many times has the tempo model been
        # called
        self.counter = 0

    def __call__(self, performed_onset, score_onset, *args, **kwargs):
        self.update_beat_period(performed_onset, score_onset, *args, **kwargs)
        self.counter += 1
        return self.beat_period, self.est_onset

    def update_beat_period(self, performed_onset, score_onset, *args, **kwargs):
        raise NotImplementedError


class ReactiveSyncModel(SyncModel):
    def __init__(self, init_beat_period=0.5, init_score_onset=0):
        super().__init__(
            init_beat_period=init_beat_period, init_score_onset=init_score_onset
        )

    def update_beat_period(self, performed_onset, score_onset):

        self.est_onset = performed_onset
        if self.prev_perf_onset:
            s_ioi = abs(score_onset - self.prev_score_onset)
            p_ioi = abs(performed_onset - self.prev_perf_onset)
            self.beat_period = p_ioi / s_ioi

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset


# Alias
RSM = ReactiveSyncModel


class MovingAverageSyncModel(SyncModel):
    def __init__(
        self,
        init_beat_period=0.5,
        init_score_onset=0,
        alpha=0.5,
        predict_onset=False,
    ):
        super().__init__(
            init_beat_period=init_beat_period, init_score_onset=init_score_onset
        )
        self.alpha = alpha
        self.predict_onset = predict_onset

    def update_beat_period(self, performed_onset, score_onset):

        if self.prev_perf_onset:
            s_ioi = abs(score_onset - self.prev_score_onset)
            p_ioi = abs(performed_onset - self.prev_perf_onset)
            beat_period = p_ioi / s_ioi

            if self.predict_onset:
                self.est_onset = self.est_onset + self.beat_period * s_ioi
            else:
                self.est_onset = performed_onset
            self.beat_period = (
                self.alpha * self.beat_period + (1 - self.alpha) * beat_period
            )
            print(self.beat_period)
        else:
            self.est_onset = performed_onset

        self.prev_score_onset = score_onset
        self.prev_perf_onset = performed_onset


# Alias
MASM = MovingAverageSyncModel


class LinearSyncModel(SyncModel):
    def __init__(
        self,
        init_beat_period=0.5,
        init_score_onset=0,
        eta_t=0.3,
        eta_p=0.7,
    ):
        super().__init__(
            init_beat_period=init_beat_period, init_score_onset=init_score_onset
        )
        self.eta_t = eta_t
        self.eta_p = eta_p

    def update_beat_period(self, performed_onset, score_onset, *args, **kwargs):

        if self.prev_perf_onset:

            s_ioi = abs(score_onset - self.prev_score_onset)
            self.est_onset = (
                self.est_onset + self.beat_period * s_ioi - self.eta_p * self.asynchrony
            )
            self.asynchrony = self.est_onset - performed_onset

        else:
            s_ioi = 0
            self.est_onset = performed_onset

        tempo_correction_term = (
            self.asynchrony if self.asynchrony != 0 and s_ioi != 0 else 0
        )
        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset

        if tempo_correction_term < 0:
            beat_period = self.beat_period - self.eta_t * tempo_correction_term
        else:
            beat_period = self.beat_period - 2 * self.eta_t * tempo_correction_term

        if beat_period > 0.25 and beat_period <= 3:
            self.beat_period = beat_period


# Alias
LSM = LinearSyncModel


class JointAdaptationAnticipationSyncModel(SyncModel):
    def __init__(
        self,
        init_beat_period=0.5,
        init_score_onset=0.0,
        alpha=0.5,
        beta=0.2,
        delta=0.8,
        gamma=0.1,
        rng_motor=np.random.RandomState(1984),
        rng_timekeeper=np.random.RandomState(1984),
    ):
        super().__init__(
            init_beat_period=init_beat_period, init_score_onset=init_score_onset
        )
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.omdelta = 1 - delta
        self.gamma = gamma

        self.score_iois = []
        self.perf_iois = []
        # Have an initial value for beat periods
        self.instant_beat_periods = [1]
        self.cnt = 0
        self.motor_noise = rng_motor
        self.timekeeper_noise = rng_timekeeper
        self.tkns = []
        self.mns = []

    def update_beat_period(self, performed_onset, score_onset):

        if self.cnt > 0:
            # perf_ioi = performed_onset - self.prev_perf_onset
            self.perf_iois.append(performed_onset - self.prev_perf_onset)
            self.score_iois.append(score_onset - self.prev_score_onset)
            ibp = self.perf_iois[-1] / self.score_iois[-1]

            if ibp >= 0.25 and ibp <= 3:
                # CC: add checks for instant beat period
                self.instant_beat_periods.append(ibp)
            else:
                self.instant_beat_periods.append(self.instant_beat_periods[-1])

            if self.cnt > 1:
                # ADAPTATION MODULE
                # TODO: add timekeeper noise
                self.ad_est_onset = (
                    self.est_onset
                    + self.beat_period * self.score_iois[-1]
                    - self.alpha * self.asynchrony
                    + self.tkns[-1]
                )
                # update beat period
                self.beat_period = self.beat_period - self.beta * (
                    self.asynchrony  # / self.score_iois[-2]
                )

                # ANTICIPATION MODULE

                # add timekeeper noise
                self.an_onset = (
                    self.prev_perf_onset
                    + (
                        self.delta * self.extrapolated_interval
                        + self.omdelta * self.preserverated_interval
                    )
                    * self.score_iois[-1]
                    + self.tkns[-1]
                )

                self.extrapolated_interval = (
                    2 * self.instant_beat_periods[-1] - self.instant_beat_periods[-2]
                )

                self.preserverated_interval = self.instant_beat_periods[-1]

                # JOINT MODULE

                self.joint_asynchrony = self.ad_est_onset - self.an_onset
                self.est_onset = (
                    self.ad_est_onset
                    - self.gamma * self.joint_asynchrony
                    + self.mns[-1]
                    - self.mns[-2]
                )

            else:
                # For the first IOI wi do not really have
                # an asynchrony yet
                self.est_onset = self.est_onset + self.beat_period * self.score_iois[-1]

                # perhaps use instant beat period instead?
                # self.beat_period = ibp
                self.extrapolated_interval = (
                    2 * self.instant_beat_periods[-1] - self.beat_period
                )
                self.preserverated_interval = self.beat_period
        else:
            self.est_onset = performed_onset

        # update global
        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset
        self.asynchrony = self.est_onset - performed_onset
        self.cnt += 1
        # Check other noise types
        self.tkns.append(self.timekeeper_noise.normal(0, 0.005))
        self.mns.append(self.motor_noise.normal(0, 0.0025))


# Alias
JADAMSM = JointAdaptationAnticipationSyncModel


class LinearTempoExpectationsSyncModel(SyncModel):
    def __init__(
        self,
        init_beat_period=0.5,
        init_score_onset=0,
        eta_t=0.2,
        eta_p=0.6,
        tempo_expectations_func=None,
    ):

        super().__init__(
            init_beat_period=init_beat_period, init_score_onset=init_score_onset
        )
        self.eta_t = eta_t
        self.eta_p = eta_p

        if tempo_expectations_func is None:
            # this is a dummy model so that it does not casue
            # an error in main.py
            self.tempo_expectations_func = lambda x: init_beat_period
        elif callable(tempo_expectations_func):
            self.tempo_expectations_func = tempo_expectations_func
        elif isinstance(tempo_expectations_func, np.ndarray):
            self.tempo_expectations_func = interp1d(
                x=tempo_expectations_func[:, 0],
                y=tempo_expectations_func[:, 1],
                kind="linear",
                fill_value="extrapolate",
            )
        else:
            raise ValueError(
                "`tempo_expectations_func` should be a " "callable or None"
            )
        self.has_tempo_expectations = True

        self.scale_factor = None

        # self.counter = 0

        self.first_score_onset = 0

        self.init_beat_period = init_beat_period

    @property
    def init_beat_period(self):
        return self._init_beat_period

    @init_beat_period.setter
    def init_beat_period(self, value):
        self._init_beat_period = value
        self.scale_factor = value / self.tempo_expectations_func(self.first_score_onset)
        print("scale_factor", self.scale_factor)
        print("te_init", self.tempo_expectations_func(self.first_score_onset))

    @property
    def first_score_onset(self):
        if self._fiso is None:
            self._fiso = self.prev_score_onset

        return self._fiso

    @first_score_onset.setter
    def first_score_onset(self, val):
        self._fiso = val

    def tempo_expectations(self, score_onset):
        # relative to the first ioi
        return float(self.tempo_expectations_func(score_onset) * self.scale_factor)

    def update_beat_period(self, performed_onset, score_onset, *args, **kwargs):

        if self.prev_perf_onset:

            s_ioi = abs(score_onset - self.prev_score_onset)
            self.est_onset = (
                self.est_onset + self.beat_period * s_ioi - self.eta_p * self.asynchrony
            )
            self.asynchrony = self.est_onset - performed_onset

        else:
            s_ioi = 0
            self.est_onset = performed_onset

        tempo_correction_term = (
            self.asynchrony if self.asynchrony != 0 and s_ioi != 0 else 0
        )
        self.prev_perf_onset = performed_onset
        self.prev_score_onset = score_onset

        expected_beat_period = self.tempo_expectations(score_onset)
        beat_period = expected_beat_period - self.eta_t * tempo_correction_term

        # bp_rel_diff = abs(beat_period - expected_beat_period)/expected_beat_period

        # if bp_rel_diff < 0.09:
        #     print('adj tempo')
        #     beat_period = expected_beat_period

        if beat_period > 0.25 and beat_period < 3:
            self.beat_period = beat_period


# Alias
LTESM = LinearTempoExpectationsSyncModel
