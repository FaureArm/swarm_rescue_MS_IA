import numpy as np

from swarm_rescue.solutions.contracts.motion import MotionReference
from swarm_rescue.solutions.contracts.state import StateEstimate


class DirectNavigator:
    """Pass the destination directly to control, preserving direct flight."""

    def navigate(
        self, destination: np.ndarray | None, state: StateEstimate,
    ) -> MotionReference | None:
        if destination is None:
            return None
        return MotionReference(destination)
