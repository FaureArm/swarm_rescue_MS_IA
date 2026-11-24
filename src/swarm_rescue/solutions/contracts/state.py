"""Estimated state in world pixels, radians and pixels per simulation step."""

from typing import NamedTuple

import numpy as np


class StateEstimate(NamedTuple):
    """Borrowed, read-only inputs to consumers; None marks unavailable fields."""

    position: np.ndarray | None
    yaw: float | None
    velocity: np.ndarray | None

    @property
    def is_available(self) -> bool:
        return (
            self.position is not None
            and self.yaw is not None
            and self.velocity is not None
        )
