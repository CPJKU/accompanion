# -*- coding: utf-8 -*-
"""
On-line Dynamic Time Warping

TODO
----
* Cythonize
"""
import numpy as np

from scipy.interpolate import interp1d

from matchmaker.utils import distances
from matchmaker.utils.distances import vdist, Metric

DEFAULT_LOCAL_COST = 'Manhattan'
WINDOW_SIZE = 50
STEP_SIZE = 5


class OnlineTimeWarping_old(object):
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
    def __init__(self, reference_features,
                 window_size=WINDOW_SIZE,
                 step_size=STEP_SIZE,
                 local_cost_fun=DEFAULT_LOCAL_COST):

        self.reference_features = reference_features
        self.input_features = []

        # Set local cost function
        if isinstance(local_cost_fun, str):
            # If local_cost_fun is a string
            self.local_cost_fun = getattr(distances, local_cost_fun)()

        elif isinstance(local_cost_fun, tuple):
            # local_cost_fun is a tuple with the arguments to instantiate
            # the cost
            self.local_cost_fun = getattr(distances,
                                          local_cost_fun[0])(
                                              **local_cost_fun[1])

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

        self.current_position = 0
        self.positions = []
        self.warping_path = []
        # For debugging
        self.windows = []

        self.global_cost_matrix = np.ones(
            (reference_features.shape[0] + 1, 2)) * np.infty
        self.input_index = 0
        self.go_backwards = False

    def __call__(self, input):
        self.step(input)
        return self.current_position

    def get_window(self, position):
        window = range(max(position - self.window_size, 0),
                       min(position + self.window_size, self.N_ref))
        return window

    @property
    def window_index(self):
        return self.current_position

    def step(self, input_features):
        """
        Step
        """

        # Do we really need to keep track of the observations?
        self.input_features.append(input_features)

        # select window
        window = self.get_window(self.window_index)
        # for debugging
        self.windows.append(window)
        min_costs = np.infty
        min_index = max(self.current_position - self.step_size, 0)

        # compute local cost beforehand as it is much faster (~twice as fast)
        window_cost = self.vdist(self.reference_features[window],
                                 input_features,
                                 self.local_cost_fun)
        # for each score position in window compute costs and check for
        # possible current score position
        for idx, score_index in enumerate(window):

            # special case: cell (0,0)
            if score_index == self.input_index == 0:
                # compute cost
                self.global_cost_matrix[1, 1] = window_cost.sum()
                min_costs = self.global_cost_matrix[1, 1]
                min_index = 0
                continue

            # get the previously computed local cost
            local_dist = window_cost[idx]

            # update global costs
            dist1 = self.global_cost_matrix[score_index, 1] + local_dist
            dist2 = self.global_cost_matrix[score_index + 1, 0] + local_dist
            dist3 = self.global_cost_matrix[score_index, 0] + local_dist

            min_dist = min(dist1, dist2, dist3)
            self.global_cost_matrix[score_index + 1, 1] = min_dist

            norm_cost = min_dist / (self.input_index + score_index + 1)

            # check if new cell has lower costs and might be current position
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        # global_cost_matrix[:, 1] *= np.infty
        # add small constant to avoid NaN
        self.global_cost_matrix[:, 1] = (self.global_cost_matrix[:, 1] + 1e-10) * \
            np.infty
        # adapt current_position: do not go backwards, but also go a maximum
        # of N steps forward
        self.current_position = min(max(self.current_position, min_index),
                                    self.current_position + self.step_size)
        # self.current_position = min(
        #     max(self.current_position - self.step_size, min_index),
        #     self.current_position + self.step_size)
        
        self.positions.append(self.current_position)
        self.warping_path.append((self.input_index,
                                  self.current_position))
        # update input index
        self.input_index += 1


# alias
OLTW = OnlineTimeWarping_old


class OnlineTimeWarping(object):
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
    ----------b
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
    def __init__(self, reference_features,
                 window_size=WINDOW_SIZE,
                 step_size=STEP_SIZE,
                 local_cost_fun=DEFAULT_LOCAL_COST):

        self.reference_features = reference_features
        self.input_features = []

        # Set local cost function
        if isinstance(local_cost_fun, str):
            # If local_cost_fun is a string
            self.local_cost_fun = getattr(distances, local_cost_fun)()

        elif isinstance(local_cost_fun, tuple):
            # local_cost_fun is a tuple with the arguments to instantiate
            # the cost
            self.local_cost_fun = getattr(distances,
                                          local_cost_fun[0])(
                                              **local_cost_fun[1])

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

        self.current_position = 0
        self.positions = []
        self.warping_path = []
        # For debugging
        self.windows = []

        self.global_cost_matrix = np.ones(
            (reference_features.shape[0] + 1, 2)) * np.infty
        self.input_index = 0
        self.go_backwards = False

    def __call__(self, input):
        self.step(input)
        return self.current_position

    def get_window(self, position):
        window = range(max(position - self.window_size, 0),
                       min(position + self.window_size, self.N_ref))
        return window

    @property
    def window_index(self):
        return self.current_position

    def step(self, input_features):
        """
        Step
        """

        # Do we really need to keep track of the observations?
        self.input_features.append(input_features)

        # select window
        window = self.get_window(self.window_index)
        # for debugging
        self.windows.append(window)
        min_costs = np.infty
        min_index = 0

        # compute local cost beforehand as it is much faster (~twice as fast)
        window_cost = self.vdist(self.reference_features[window],
                                 input_features,
                                 self.local_cost_fun)
        # for each score position in window compute costs and check for
        # possible current score position
        for idx, score_index in enumerate(window):

            # special case: cell (0,0)
            if score_index == self.input_index == 0:
                # compute cost
                self.global_cost_matrix[1, 1] = window_cost.sum()
                min_costs = self.global_cost_matrix[1, 1]
                min_index = 0
                continue

            # get the previously computed local cost
            local_dist = window_cost[idx]

            # update global costs
            dist1 = self.global_cost_matrix[score_index, 1] + local_dist
            dist2 = self.global_cost_matrix[score_index + 1, 0] + local_dist
            dist3 = self.global_cost_matrix[score_index, 0] + local_dist

            min_dist = min(dist1, dist2, dist3)
            self.global_cost_matrix[score_index + 1, 1] = min_dist

            norm_cost = min_dist / (self.input_index + score_index + 1)

            # check if new cell has lower costs and might be current position
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        # global_cost_matrix[:, 1] *= np.infty
        # add small constant to avoid NaN
        self.global_cost_matrix[:, 1] = (self.global_cost_matrix[:, 1] + 1e-10) * \
            np.infty
        # adapt current_position: do not go backwards, but also go a maximum
        # of N steps forward
        # self.current_position = min(max(self.current_position, min_index),
        #                             self.current_position + self.step_size)
        self.current_position = min(
            max(self.current_position - self.step_size, min_index),
            self.current_position + self.step_size)
        
        self.positions.append(self.current_position)
        self.warping_path.append((self.input_index,
                                  self.current_position))
        # update input index
        self.input_index += 1



class TempoOnlineTimeWarping(OnlineTimeWarping):
    """
    Step

    Note
    ----
    For symbolic alignment, it is important that the first window is "not empty",
    relevant start of the input signal (e.g., the first MIDI message for symbolic).
    For now, this method would require some modifications to work with audio, since
    it would need to "know" the starting time.
    """
    def __init__(self, reference_features,
                 tempo_model,
                 reference_positions=None,
                 # polling_period,
                 window_size=WINDOW_SIZE,
                 step_size=STEP_SIZE,
                 local_cost_fun=DEFAULT_LOCAL_COST):
        super().__init__(reference_features=reference_features,
                         window_size=window_size,
                         step_size=step_size,
                         local_cost_fun=local_cost_fun)
        self.tempo_model = tempo_model

        if reference_positions is None:
            # setup time map
            reference_positions = np.arange(self.N_ref).astype(float)
        else:
            if len(reference_positions) != self.N_ref:
                raise ValueError("`reference_positions` should have the same "
                                 "length as `reference_features`")
            reference_positions = reference_positions
                                 

        tempi = self.tm(reference_positions)

        expected_positions = np.r_[0, np.cumsum(tempi[:-1])]

        self.score_index_map = interp1d(expected_positions, np.arange(self.N_ref),
                                        bounds_error=False,
                                        fill_value=(0, self.N_ref))

    def get_window(self, position):
        window_center = int(np.round(self.score_index_map(position)))
        window = range(max(window_center - self.window_size, 0),
                       min(window_center + self.window_size, self.N_ref))
        return window

    @property
    def window_index(self):
        return self.input_index

    def step(self, input_features):
        """
        Step
        """

        # Do we really need to keep track of the observations?
        self.input_features.append(input_features)

        # select window
        window = self.get_window(self.window_index)
        # for debugging
        self.windows.append(window)
        min_costs = np.infty
        min_index = 0

        # compute local cost beforehand as it is much faster (~twice as fast)
        window_cost = self.vdist(self.reference_features[window],
                                 input_features,
                                 self.local_cost_fun)
        # for each score position in window compute costs and check for
        # possible current score position
        for idx, score_index in enumerate(window):

            # special case: cell (0,0)
            if score_index == self.input_index == 0:
                # compute cost
                self.global_cost_matrix[1, 1] = window_cost.sum()
                min_costs = self.global_cost_matrix[1, 1]
                min_index = 0
                continue

            # get the previously computed local cost
            local_dist = window_cost[idx]

            # update global costs
            dist1 = self.global_cost_matrix[score_index, 1] + local_dist
            dist2 = self.global_cost_matrix[score_index + 1, 0] + local_dist
            dist3 = self.global_cost_matrix[score_index, 0] + local_dist

            min_dist = min(dist1, dist2, dist3)
            self.global_cost_matrix[score_index + 1, 1] = min_dist

            norm_cost = min_dist / (self.input_index + score_index + 1)

            # check if new cell has lower costs and might be current position
            if norm_cost < min_costs:
                min_costs = norm_cost
                min_index = score_index

        self.global_cost_matrix[:, 0] = self.global_cost_matrix[:, 1]
        # global_cost_matrix[:, 1] *= np.infty
        # add small constant to avoid NaN
        self.global_cost_matrix[:, 1] = (self.global_cost_matrix[:, 1] + 1e-10) * \
            np.infty
        # adapt current_position: do not go backwards, but also go a maximum
        # of N steps forward
        self.current_position = min(max(self.current_position, min_index),
                                    self.current_position + self.step_size)

        self.positions.append(self.current_position)
        self.warping_path.append((self.input_index,
                                  self.current_position))
        # update input index
        self.input_index += 1



if __name__ == '__main__':

    pass
