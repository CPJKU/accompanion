# -*- coding: utf-8 -*-
"""
This module implements the specific HMM for the score follower for Accompanion.
"""
import numpy as np

from accompanion.mtchmkr.base import OnlineAlignment

class ObservationModel(OnlineAlignment):
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
        # Store the parameters of the object:
        self._pitch_profiles = pitch_profiles
        self._ioi_matrix = ioi_matrix
        self._ioi_precision = ioi_precision
        # Compute the IOI normalization term:
        self._ioi_norm_term = np.sqrt(0.5 * self._ioi_precision / np.pi)

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


class ScoreHMM(object):
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

    _initial_distribution : numpy array
        The initial distribution of the model. If not given, it is asumed to
        be uniform.

    _forward_variable : numpy array
        The current (latest) value of the forward variable.

    _variation_coeff : float
        The normalized coefficient of variation of the current (latest) forward
        variable. Used to determine the confidence of the prediction of the HMM.

    _current_state : int
        The index of the current state of the HMM.
    """

    def __init__(
        self,
        transition_matrix,
        pitch_profiles,
        ioi_matrix,
        ioi_precision,
        score_onsets,
        initial_distribution=None,
    ):
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
        # Check if the transition probability matrix is a square matrix:
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            # Raise invalid input exception:
            raise ValueError("Invalid Transition Probability Matrix (not square).")
        # Initialize the transition model:
        self._transition_matrix = transition_matrix
        # Initialize the observation model:
        self._observation_model = ObservationModel(
            pitch_profiles, ioi_matrix, ioi_precision
        )

        self._score_onsets = score_onsets

        # Define the initial probability:
        if initial_distribution is not None:
            # Check if the initial_distribution is valid:
            if len(initial_distribution) != (transition_matrix.shape[0]):
                # Raise an error:
                raise ValueError(
                    "Invalid Initial Distribution size. \
                Initial Distribution size: {ids}, \
                Transition Matrix Dimension: {tmd}".format(
                        ids=len(initial_distribution), tmd=transition_matrix.shape[0]
                    )
                )
            # If provided:
            self._initial_distribution = initial_distribution
        else:
            # Create a uniform distribution:
            self._initial_distribution = (
                np.ones(transition_matrix.shape[0]) / transition_matrix.shape[0]
            )

        # Declare the first forward variable from the initial distribution:
        self._forward_variable = self._initial_distribution

        # Initialize the variable to store the current state of the HMM:
        self._current_state = np.argmax(self._forward_variable)

        # Variable to show if this is the first run of the forward algorithm:
        self._first_time = True

    def forward_algorithm_online(self, pitch_obs, ioi_obs, period_est):
        """
        Performs the online version (a single step from a single observation) of
        the forward algorithm for the Score HMM. Updates the forward variable
        and the current state of the HMM for the latest observed event.

        Parameters
        ----------
        pitch_obs : array-like
            The performed pitches in the observed event.

        ioi_obs : float
            The value of the observed ioi between the last two performed notes
            in seconds.

        period_est : float
            The value of the estimated period by the Linear model in the
            IOIObservationManager object. Used in the estimation of IOIs in the
            performance in seconds.
        """
        # Compute the pitch and IOI probabilities separately:
        pitch_prob = self._observation_model.compute_pitch_observation_probability(
            pitch_obs
        )

        ioi_prob = self._observation_model.compute_ioi_observation_probability(
            ioi_obs, self._current_state, period_est
        )

        # Check if we are still in the 0th state:
        if self._first_time:
            # Just give the forward variable as transition prob:
            transition_prob = self._forward_variable
            # Run the computation methods of the transition and observation
            # models:
            self._forward_variable = (pitch_prob * ioi_prob) * transition_prob
            # Reset the first time:
            self._first_time = False

        else:
            # Compute the transition probabilities:
            transition_prob = np.dot(self._transition_matrix.T, self._forward_variable)
            # Run the computation methods of the transition and observation
            # models:
            self._forward_variable = (pitch_prob * ioi_prob) * transition_prob

        # Normalize the newly computed forward variable:
        self._forward_variable /= self._forward_variable.sum()

        # Update the current state:
        self._current_state = np.argmax(self._forward_variable)

    def __call__(self, observations):
        pitch_observations, ioi_observations, period_estimation = observations

        self.forward_algorithm_online(
            pitch_observations, ioi_observations, period_estimation
        )

        return self._score_onsets[self._current_state]

    @property
    def forward_variable(self):
        """
        Get the latest forward variable of the HMM.

        Returns
        -------
        forward_variable : numpy array
            The current (latest) value of the forward variable.
        """
        return self._forward_variable

    @property
    def initial_distribution(self):
        """
        Get the initial distribution of the HMM.

        Returns
        -------
        initial_distribution : numpy array
            The initial probability distribution of the states in the HMM.
        """
        return self._initial_distribution

    @property
    def current_state(self):
        """
        Get the current state of the HMM.

        Returns
        -------
        current_state : int
            The current state of the HMM.
        """
        return self._current_state
