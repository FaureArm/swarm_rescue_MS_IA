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
    def __init__(self, width = 2000, height = 2000, pos_hit_increment = 0.5, neg_hit_increment = 0.5):
        self.width = width + 100 # padding for values outside due to noise
        self.height = height + 100

        # arrays holding our values
        self.log_odds = np.zeros((self.width, self.height), dtype=np.float32)
        self.last_observed = np.full((self.width, self.height), -np.inf, dtype=np.float64)
        
        self.log_odd_plus = pos_hit_increment
        self.log_odd_minus = neg_hit_increment

    def update_log_odds(self, x, y, pos_hit):
        if pos_hit:
            self.log_odds[self.width//2 + x, self.height//2 + y] += self.log_odd_plus
        else:
            self.log_odds[self.width//2 + x, self.height//2 + y] -= self.log_odd_minus
        return True
