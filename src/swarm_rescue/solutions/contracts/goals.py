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
