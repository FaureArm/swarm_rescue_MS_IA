"""Navigation references and body-frame actuator commands."""

from typing import NamedTuple

import numpy as np


class MotionReference(NamedTuple):
    """A borrowed, read-only destination [x, y] in world pixels."""

    destination: np.ndarray


class MotionCommand(NamedTuple):
    """Forward/lateral thrust and rotation, each in [-1, 1]."""

    forward: float
    lateral: float
    rotation: float
