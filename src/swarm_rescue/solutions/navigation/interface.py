from typing import Protocol

import numpy as np

from swarm_rescue.solutions.contracts.motion import MotionReference
from swarm_rescue.solutions.contracts.state import StateEstimate


class Navigator(Protocol):
    """Translate a world goal into a motion reference; None requests a stop.

    Arrival is currently evaluated by the behavior. This contract does not
    imply obstacle checking or blocked-goal detection.
    """

    def navigate(
        self, destination: np.ndarray | None, state: StateEstimate,
    ) -> MotionReference | None: ...
