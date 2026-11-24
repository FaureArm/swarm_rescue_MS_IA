"""Sensor inputs in world pixels, radians and simulation-step units.

None means that a measurement is unavailable. Arrays are borrowed for the
current tick; consumers must copy them if they retain or modify their contents.
"""

from typing import NamedTuple

import numpy as np


class StateObservation(NamedTuple):
    position: np.ndarray | None
    yaw: float | None
    velocity: np.ndarray | None
    timestep: int
    angular_velocity: float | None = None


class LidarObservation(NamedTuple):
    position: np.ndarray
    yaw: float
    ranges: np.ndarray | None
    angles: np.ndarray | None
    timestep: int
