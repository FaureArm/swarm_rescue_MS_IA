"""Shared data structures for the drone's components."""

from typing import NamedTuple

import numpy as np


class MissionDecision(NamedTuple):
    # small class to hold decisions and reuse easily
    destination: np.ndarray | None
    grasper: int


class Waypoint(NamedTuple):
    """A position in the map's world coordinates, in pixels
    Attributes:
        x (float): The x-coordinate of the waypoint.
        y (float): The y-coordinate of the waypoint.
    TODO: Add a heading attribute to the waypoint, to allow for more precise navigation
    """

    x: float
    y: float


class OccupancyGrid:
    """class to hold grid occupancy probabilistic map.
    probability are loggits, so a value of 0 for the log_odds att means 50% proba
    TODO: Initialise with max map size. 
    add getters and setters ?
    """

    MIN_LOG_ODDS = -5.0
    MAX_LOG_ODDS = 5.0

    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    UNCERTAIN = 3

    FREE_THRESHOLD = -0.1
    OCCUPIED_THRESHOLD = 0.1

    def __init__(self, width = 2000, height = 2000, pos_hit_increment = 0.5, neg_hit_increment = 0.5):
        self.width = width + 100 # padding for values outside due to noise
        self.height = height + 100

        # arrays holding our values
        self.log_odds = np.zeros((self.width, self.height), dtype=np.float32)
        self.last_observed = np.full((self.width, self.height), -np.inf, dtype=np.float64)
        
        self.log_odd_plus = pos_hit_increment
        self.log_odd_minus = neg_hit_increment

    def update_log_odds(self, x, y, pos_hit, timestep):
        """Update evidence and observation time for valid cells."""
        if not np.isfinite(timestep) or timestep < 0:
            raise ValueError("Observation timestep must be finite and nonnegative")

        grid_x, grid_y, in_bounds = self.world_to_grid(x, y)

        if not np.any(in_bounds):
            return False

        indices = grid_x[in_bounds], grid_y[in_bounds]
        increment = self.log_odd_plus if pos_hit else -self.log_odd_minus

        np.add.at(self.log_odds, indices, increment)

        self.log_odds[indices] = np.clip(
            self.log_odds[indices],
            self.MIN_LOG_ODDS,
            self.MAX_LOG_ODDS,
        )
        self.last_observed[indices] = timestep

        return True

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid indices and an in-bounds mask.

        Invalid entries receive placeholder indices (0, 0).
        """
        world_x, world_y = np.broadcast_arrays(
            np.asarray(x, dtype=float),
            np.asarray(y, dtype=float),
        )

        grid_x = np.rint(world_x) + self.width // 2
        grid_y = np.rint(world_y) + self.height // 2

        in_bounds = (
            np.isfinite(grid_x)
            & np.isfinite(grid_y)
            & (grid_x >= 0)
            & (grid_x < self.width)
            & (grid_y >= 0)
            & (grid_y < self.height)
        )

        grid_x = np.where(in_bounds, grid_x, 0).astype(np.intp)
        grid_y = np.where(in_bounds, grid_y, 0).astype(np.intp)

        return grid_x, grid_y, in_bounds

    def classify_cells(self):
        """Label unseen cells UNKNOWN and observed, ambiguous cells UNCERTAIN."""
        states = np.full(
            self.log_odds.shape,
            self.UNKNOWN,
            dtype=np.uint8,
        )

        observed = np.isfinite(self.last_observed)
        states[observed] = self.UNCERTAIN
        states[observed & (self.log_odds < self.FREE_THRESHOLD)] = self.FREE
        states[observed & (self.log_odds > self.OCCUPIED_THRESHOLD)] = self.OCCUPIED

        return states


    def frontier_mask(self):
        """Return unknown cells adjacent to free space using four neighbors."""
        states = self.classify_cells()

        # create boolean arrays of same size as occupancy grid
        free = states == self.FREE
        unknown = states == self.UNKNOWN

        # create an all-False boolean array with the same shape as the grid
        has_free_neighbor = np.zeros_like(free)

        # boolean mask of neighbors along x
        has_free_neighbor[1:, :] |= free[:-1, :] # x+1
        has_free_neighbor[:-1, :] |= free[1:, :] # x-1

        #  boolean mask of neighbors along y
        has_free_neighbor[:, 1:] |= free[:, :-1] #y+1
        has_free_neighbor[:, :-1] |= free[:, 1:] #y-1

        # rreturn a boolean mask where true means frontier cell
        return unknown & has_free_neighbor
