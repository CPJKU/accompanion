# -*- coding: utf-8 -*-
"""
Module for all utility math for the transition parts of the HMM.
"""
# Native Python Library:
import logging

# Third Party Packages:
import numpy as np
from scipy.stats import gumbel_l

# Set the LOGGER of this module:
LOGGER = logging.getLogger(__name__)


def gumbel_transition_model(
    n_states,
    mp_trans_state=1,
    scale=0.5,
    inserted_states=True,
):
    """
    Compute a transiton matrix, where each row follows a normalized Gumbel
    distribution.

    Parameters
    ----------
    n_states : int
        The number of states in the Hidden Markov Model (HMM), which is required
        for the size of the matrix.

    mp_trans_state : int
        Which state should have the largest probability to be transitioned into
        from the current state the model is in.
        Default = 1, which means that the model would prioritize transitioning
        into the state that is next in line, e.g. from State 3 to State 4.

    scale : float
        The scale parameter of the distribution.
        Default = 0.5

    inserted_states : boolean
        Indicates whether the HMM includes inserted states (intermediary states
        between chords for errors and insertions in the score following).
        Default = True

    Returns
    -------
    transition_matrix : numpy array
        The computed transition matrix for the HMM.
    """
    # Initialize transition matrix:
    transition_matrix = np.zeros((n_states, n_states), dtype="f8")

    # Compute transition matrix:
    for i in range(n_states):
        if inserted_states:
            if np.mod(i, 2) == 0:
                transition_matrix[i] = gumbel_l.pdf(
                    np.arange(n_states), loc=i + mp_trans_state * 2, scale=scale
                )
            else:
                transition_matrix[i] = gumbel_l.pdf(
                    np.arange(n_states), loc=i + mp_trans_state * 2 - 1, scale=scale
                )
        else:
            transition_matrix[i] = gumbel_l.pdf(
                np.arange(n_states), loc=i + mp_trans_state * 2 - 1, scale=scale
            )

    # Normalize transition matrix (so that it is a proper stochastic matrix):
    transition_matrix /= transition_matrix.sum(1, keepdims=True)

    # Return the computed transition matrix:
    return transition_matrix


def gumbel_init_dist(config, n_states):
    """
    Compute the initial probabilites for all states in the Hidden Markov Model
    (HMM), which follow a Gumbel distribution.

    Parameters
    ----------
    n_states : int
        The number of states in the Hidden Markov Model (HMM), which is required
        for the size of the initial probabilites vector.

    Returns
    -------
    init_dist : numpy array
        The computed initial probabilities in the form of a vector.
    """
    # Construct the initial probability. First, extract the parameters:
    try:
        # initial dist pars: [loc, scale]
        init_dist_pars = config["initial_distribution"]
        # Create and normalize the distribution:
        init_dist = gumbel_l.pdf(
            np.arange(n_states), loc=init_dist_pars[0], scale=init_dist_pars[1]
        )
        init_dist /= init_dist.sum()

    except KeyError():
        LOGGER.warning(
            "Initial distribution parameters not found in config \
        dictionary. Setting to default: uniform"
        )
        # Create the uniform distribution:
        init_dist = np.ones(n_states) / n_states
        init_dist /= init_dist.sum()

    # Return the distribution:
    return init_dist


def compute_pitch_profiles(
    chord_pitches,
    profile=np.array([0.02, 0.02, 1, 0.02, 0.02]),
    eps=0.01,
    piano_range=False,
    normalize=True,
    inserted_states=True,
):
    """
    Pre-compute the pitch profiles used in calculating the pitch
    observation probabilities.

    Parameters
    ----------
    chord_pitches : array-like
        The pitches of each chord in the piece.

    profile : numpy array
        The probability "gain" of how probable are the closest pitches to
        the one in question.

    eps : float
        The epsilon value to be added to each pre-domputed pitch profile.

    piano_range : boolean
        Indicates whether the possible MIDI pitches are to be restricted
        within the range of a piano.

    normalize : boolean
        Indicates whether the pitch profiles are to be normalized.

    inserted_states : boolean
        Indicates whether the HMM uses inserted states between chord states.

    Returns
    -------
    pitch_profiles : numpy array
        The pre-computed pitch profiles.
    """
    # Compute the high and low contexts:
    low_context = profile.argmax()
    high_context = len(profile) - profile.argmax()

    # Get the number of states, based on the presence of inserted states:
    if inserted_states:
        n_states = 2 * len(chord_pitches) - 1
    else:
        n_states = len(chord_pitches)
    # Initialize the numpy array to store the pitch profiles:
    pitch_profiles = np.zeros((n_states, 128))

    # Compute the profiles:
    for i in range(n_states):
        # Work on chord states (even indices), not inserted (odd indices):
        if np.mod(i, 2) == 0:
            chord = chord_pitches[i // 2]
            for pitch in chord:
                lowest_pitch = pitch - low_context
                highest_pitch = pitch + high_context
                # Compute the indices which are to be updated:
                idx = slice(np.maximum(lowest_pitch, 0), np.minimum(highest_pitch, 128))
                # Add the values:
                pitch_profiles[i, idx] += profile

        # Add the extra value:
        pitch_profiles[i] += eps

    # Check whether to trim and normalize:
    if piano_range:
        pitch_profiles = pitch_profiles[:, 21:109]
    if normalize:
        pitch_profiles /= pitch_profiles.sum(1, keepdims=True)

    # Return the profiles:
    return pitch_profiles
