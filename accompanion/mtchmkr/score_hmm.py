# -*- coding: utf-8 -*-
"""
This module implements the specific HMM for the score follower for Accompanion.
"""
from typing import Optional

import numpy as np
from hiddenmarkov import ObservationModel, ConstantTransitionModel, HiddenMarkovModel

from accompanion.mtchmkr.base import OnlineAlignment

from accompanion.accompanist.tempo_models import SyncModel


class PitchIOIObservationModel(ObservationModel):
    """
    Computes the probabilities that an observation was emitted, i.e. the
    likelihood of observing performed notes at the current moment/state.

    Parameters
    ----------
    _pitch_profiles : numpy array
        he pre-computed pitch profiles, for each separate possible pitch
        in the MIDI range. Used in calculating the pitch observation
        probabilities.

    _ioi_matrix : numpy array
        The pre-computed score IOI values in beats, from each unique state
        to all other states, stored in a matrix.

    _ioi_precision : float
        The precision parameter for computing the IOI observation probability.

    _ioi_norm_term : float
        The normalization term of the Gaussian distribution used for the
        computation of the IOI probabilities.

    TODO
    ----
    * Implement log probabilities
    """

    def __init__(self, pitch_profiles, ioi_matrix, ioi_precision):
        """
        The initialization method.

        Parameters
        ----------
        pitch_profiles : numpy array
            he pre-computed pitch profiles, for each separate possible pitch
            in the MIDI range. Used in calculating the pitch observation
            probabilities.

        ioi_matrix : numpy array
            The pre-computed score IOI values in beats, from each unique state
            to all other states, stored in a matrix.

        ioi_precision : float
            The precision parameter for computing the IOI observation
            probability.
        """
        super().__init__(use_log_probabilities=False)
        # Store the parameters of the object:
        self._pitch_profiles = pitch_profiles
        self._ioi_matrix = ioi_matrix
        self._ioi_precision = ioi_precision
        # Compute the IOI normalization term:
        self._ioi_norm_term = np.sqrt(0.5 * self._ioi_precision / np.pi)
        self.current_state = None

    def compute_pitch_observation_probability(self, pitch_obs):
        """
        Compute the pitch observation probability.

        Parameters
        ----------
        pitch_obs : numpy array
            All the MIDI pitch values in an observation.

        Returns
        -------
        pitch_obs_prob : numpy array
            The computed pitch observation probabilities for all states.
        """
        # Use Bernouli distribution to compute the prob:
        # Binary piano-roll observation:
        pitch_prof_obs = np.zeros((1, 128))
        pitch_prof_obs[0, pitch_obs.astype(np.int)] = 1

        # Compute Bernoulli probability:
        pitch_prob = (self._pitch_profiles ** pitch_prof_obs) * (
            (1 - self._pitch_profiles) ** (1 - pitch_prof_obs)
        )

        # Return the values:
        return np.prod(pitch_prob, 1)

    def compute_ioi_observation_probability(self, ioi_obs, current_state, tempo_est):
        """
        Compute the IOI observation probability.

        Parameters
        ----------
        ioi_obs : numpy array
            All the observed IOI.

        current_state : int
            The current state of the Score HMM.

        tempo_est : float
            The tempo estimation.

        Returns
        -------
        ioi_obs_prob : numpy array
            The computed IOI observation probabilities for each state.
        """
        # Use Gaussian distribution:
        ioi_idx = current_state if current_state is not None else 0
        # Compute the expected argument:
        exp_arg = (
            -0.5
            * ((tempo_est * self._ioi_matrix[ioi_idx] - ioi_obs[-1]) ** 2)
            * self._ioi_precision
        )

        # Return the value:
        return self._ioi_norm_term * np.exp(exp_arg)

    def get_score_ioi(self, current_state):
        """
        Get the score inter onset interval (IOI) between the current state and
        the previous state in beats.

        Parameters
        ----------
        current_state : int
            The current state of the Score HMM.

        Returns
        -------
        state_ioi : numpy.int
            The IOI in beats.
        """
        # Return the specific value:
        return self._ioi_matrix[current_state]

    def __call__(self, observation):
        pitch_obs, ioi_obs, tempo_est = observation
        observation_prob = self.compute_pitch_observation_probability(
            pitch_obs
        ) * self.compute_ioi_observation_probability(
            ioi_obs=ioi_obs, current_state=self.current_state, tempo_est=tempo_est
        )
        return observation_prob


class PitchIOIHMM(HiddenMarkovModel, OnlineAlignment):
    """
    Implements the bahaviour of a HiddenMarkovModel, specifically designed for
    the task of score following.

    Parameters
    ----------
    _transition_matrix : numpy.ndarray
        Matrix for computations of state transitions within the HMM.

    _observation_model : ObservationModel
        Object responsible for computing the observation probabilities for each
        state of the HMM.

    initial_distribution : numpy array
        The initial distribution of the model. If not given, it is asumed to
        be uniform.

    forward_variable : numpy array
        The current (latest) value of the forward variable.

    _variation_coeff : float
        The normalized coefficient of variation of the current (latest) forward
        variable. Used to determine the confidence of the prediction of the HMM.

    current_state : int
        The index of the current state of the HMM.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        pitch_profiles: np.ndarray,
        ioi_matrix: np.ndarray,
        score_onsets: np.ndarray,
        tempo_model: SyncModel,
        ioi_precision: float = 1,
        initial_probabilities: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        transition_matrix : numpy array
            The Tranistion probability matrix of HMM.

        pitch_profiles : numpy array
            The pre-computed pitch profiles, for each separate possible pitch
            in the MIDI range. Used in calculating the pitch observation
            probabilities.

        ioi_matrix : numpy array
            The pre-computed score IOI values in beats, from each unique state
            to all other states, stored in a matrix.

        ioi_precision : float
            The precision parameter for computing the IOI observation
            probability.

        score_onsets : numpy array
            TODO

        initial_distribution : numpy array
            The initial distribution of the model. If not given, it is asumed to
            be uniform.
            Default = None.
        """
        reference_features = (transition_matrix, pitch_profiles, ioi_matrix)

        observation_model = PitchIOIObservationModel(
            pitch_profiles=pitch_profiles,
            ioi_matrix=ioi_matrix,
            ioi_precision=ioi_precision,
        )

        super().__init__(
            observation_model=observation_model,
            transition_model=ConstantTransitionModel(
                transition_probabilities=transition_matrix,
                init_probabilities=initial_probabilities,
            ),
            state_space=score_onsets,
            reference_features=reference_features,
        )

        self.tempo_model = tempo_model

    def __call__(self, input):

        self.current_state = self.forward_algorithm_step(
            observation=input + (self.tempo_model.beat_period,),
            log_probabilities=False
        )

        return self.state_space[self.current_state]

    @property
    def current_state(self):
        return self.observation_model.current_state

    @current_state.setter
    def current_state(self, state):
        self.observation_model.current_state = state
