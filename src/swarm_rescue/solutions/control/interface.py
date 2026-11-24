from typing import Protocol

from swarm_rescue.solutions.contracts.motion import MotionCommand, MotionReference
from swarm_rescue.solutions.contracts.state import StateEstimate


class MotionController(Protocol):
    """Compute bounded body-frame commands from an available estimated state.

    Grasper intent is applied separately by the simulator adapter. Stateful
    controllers are constructed afresh for each drone.
    """

    def compute_commands(
        self, reference: MotionReference, estimate: StateEstimate,
    ) -> MotionCommand: ...
