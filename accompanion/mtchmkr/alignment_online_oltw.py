# -*- coding: utf-8 -*-
"""
On-line Dynamic Time Warping
"""
from typing import Callable, List, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from accompanion.mtchmkr import distances
from accompanion.mtchmkr.base import OnlineAlignment
from accompanion.mtchmkr.distances import Metric, vdist
from accompanion.mtchmkr.dtw_loop import dtw_loop, reset_cost_matrix

DEFAULT_LOCAL_COST: str = "Manhattan"
WINDOW_SIZE: int = 100
STEP_SIZE: int = 5
START_WINDOW_SIZE: int = 60


class OnlineTimeWarping(OnlineAlignment):
    """
    Fast On-line Time Warping

    Parameters
    ----------
    reference_features : np.ndarray
        A 2D array with dimensions (n_timesteps, n_features) containing the
        features of the reference the input is going to be aligned to.
    window_size : int
        Size of the window for searching the optimal path in the cumulative
        cost matrix
    step_size : int
        Size of the step
    local_cost_fun : Union[str, Callable]
        Local metric for computing pairwise distances.
    start_window_size: int
        Size of the starting window size
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

    local_cost_fun: Callable[[NDArray[np.float64]], NDArray[np.float64]]

    def __init__(
        self,
        reference_features: NDArray[np.float64],
        window_size: int = WINDOW_SIZE,
        step_size: int = STEP_SIZE,
        local_cost_fun: Union[str, Callable] = DEFAULT_LOCAL_COST,
        start_window_size: int = START_WINDOW_SIZE,
    ) -> None:
        super().__init__(reference_features=reference_features)
        # self.reference_features = reference_features
        self.input_features: List[NDArray[np.float64]] = []

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

        self.N_ref: int = self.reference_features.shape[0]
        self.window_size: int = window_size
        self.step_size: int = step_size
        self.start_window_size: int = start_window_size
        self.current_position: int = 0
        self.positions: List[int] = []
        self.warping_path: List = []
        self.global_cost_matrix: NDArray[np.float64] = (
            np.ones((reference_features.shape[0] + 1, 2)) * np.infty
        )
        self.input_index: int = 0
        self.go_backwards: bool = False
        self.update_window_index: bool = False
        self.restart: bool = False

    def __call__(self, input: NDArray[np.float64]) -> int:
        self.step(input)
        return self.current_position

    def get_window(self) -> Tuple[int, int]:
        w_size = self.window_size
        if self.window_index < self.start_window_size:
            w_size = self.start_window_size
        window_start = max(self.window_index - w_size, 0)
        window_end = min(self.window_index + w_size, self.N_ref)
        return window_start, window_end

    @property
    def window_index(self) -> int:
        return self.current_position

    def step(self, input_features: NDArray[np.float64]) -> None:
        """
        Update the current position and the warping path.
        """
        min_costs = np.infty
        min_index = max(self.window_index - self.step_size, 0)

        window_start, window_end = self.get_window()
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


if __name__ == "__main__":
    pass
