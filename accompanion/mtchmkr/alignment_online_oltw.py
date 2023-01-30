# -*- coding: utf-8 -*-
"""
On-line Dynamic Time Warping
"""
import numpy as np

from accompanion.mtchmkr.base import OnlineAlignment
from accompanion.mtchmkr import distances
from accompanion.mtchmkr.distances import vdist, Metric
from accompanion.mtchmkr.dtw_loop import (
    dtw_loop,
    reset_cost_matrix,
)

DEFAULT_LOCAL_COST = "Manhattan"
WINDOW_SIZE = 100
STEP_SIZE = 5
START_WINDOW_SIZE = 60


class OnlineTimeWarping(OnlineAlignment):
    """
    Fast On-line Time Warping

    Parameters
    ----------
    reference_features : np.ndarray
        A 2D array with dimensions (n_timesteps, n_features) containing the
        features of the reference the input is going to be aligned to.
    window_size : int (optional)
        Size of the window for searching the optimal path in the cumulative
        cost matrix
    step_size : int (optional)
        Size of the step

    Attributes
    ----------
    reference_features : np.ndarray
    window_size : int
    step_size : int
    input_features : list
        List with the input features (updates every time there is a step)
    current_position : int
        Index of the current position
    warping_path : list
        List of tuples containing the current position and the corresponding
        index in the array of `reference_features`.
    positions : list
        List of the positions for each input.
    """

    def __init__(
        self,
        reference_features,
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        local_cost_fun=DEFAULT_LOCAL_COST,
        start_window_size=START_WINDOW_SIZE,
    ):

        super().__init__(reference_features=reference_features)
        # self.reference_features = reference_features
        self.input_features = []

        # Set local cost function
        if isinstance(local_cost_fun, str):
            # If local_cost_fun is a string
            self.local_cost_fun = getattr(distances, local_cost_fun)()

        elif isinstance(local_cost_fun, tuple):
            # local_cost_fun is a tuple with the arguments to instantiate
            # the cost
            self.local_cost_fun = getattr(distances, local_cost_fun[0])(
                **local_cost_fun[1]
            )

        elif callable(local_cost_fun):
            # If the local cost is a callable
            self.local_cost_fun = local_cost_fun

        # A callable to compute the distance between the rows of matrix and a vector
        if isinstance(self.local_cost_fun, Metric):
            self.vdist = vdist
        else:
            self.vdist = lambda X, y, lcf: lcf(X, y)

        self.N_ref = self.reference_features.shape[0]
        self.window_size = window_size
        self.step_size = step_size
        self.start_window_size = start_window_size

        self.current_position = 0
        self.positions = []
        self.warping_path = []
        self.global_cost_matrix = (
            np.ones((reference_features.shape[0] + 1, 2)) * np.infty
        )
        self.input_index = 0
        self.go_backwards = False
        self.update_window_index = False
        self.restart = False

    def __call__(self, input):
        self.step(input)
        return self.current_position

    def get_window(self):
        w_size = self.window_size
        if self.window_index < self.start_window_size:
            w_size = self.start_window_size
        window_start = max(self.window_index - w_size, 0)
        window_end = min(self.window_index + w_size, self.N_ref)
        return window_start, window_end

    @property
    def window_index(self):
        return self.current_position

    def step(self, input_features):
        """
        Update the current position and the warping path.
        """
        min_costs = np.infty
        min_index = max(self.window_index - self.step_size, 0)

        window_start, window_end = self.get_window()
        # window_start = max(self.window_index - self.window_size, 0)
        # window_end = min(self.window_index + self.window_size, self.N_ref)
        # compute local cost beforehand as it is much faster (~twice as fast)
        window_cost = self.vdist(
            self.reference_features[window_start:window_end],
            input_features,
            self.local_cost_fun,
        )
        if self.restart:
            self.global_cost_matrix = reset_cost_matrix(
                global_cost_matrix=self.global_cost_matrix,
                window_cost=window_cost,
                score_index=window_start,
                N=self.N_ref + 1,
            )
            self.restart = False

        self.global_cost_matrix, min_index, min_costs = dtw_loop(
            global_cost_matrix=self.global_cost_matrix,
            window_cost=window_cost,
            window_start=window_start,
            window_end=window_end,
            input_index=self.input_index,
            min_costs=min_costs,
            min_index=min_index,
        )

        # adapt current_position: do not go backwards,
        # but also go a maximum of N steps forward
        self.current_position = min(
            max(self.current_position, min_index),
            self.current_position + self.step_size,
        )

        # update input index
        self.input_index += 1

    # # TODO review that update_position method is not needed.
    # def update_position(self, input_features, position):
    #     """
    #     Restart following from a new position.
    #     This method "forgets" the pasts and starts from
    #     scratch form a new position.
    #     """
    #     self.current_position = int(position)
    #     window_start, window_end = self.get_window()
    #     window_cost = self.vdist(
    #         self.reference_features[window_start:window_end],
    #         input_features,
    #         self.local_cost_fun
    #     )


if __name__ == "__main__":

    pass
