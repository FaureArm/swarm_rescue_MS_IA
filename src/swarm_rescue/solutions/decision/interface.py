from typing import Protocol

import numpy as np

from swarm_rescue.solutions.contracts.goals import MissionDecision


class Behavior(Protocol):
    """Choose a goal and action intent from the current world position.

    A None destination requests a stop. Arrival/reselection remains owned by
    the behavior. Each drone gets its own behavior instance from assembly.
    """

    @property
    def goal(self) -> np.ndarray | None: ...

    def decide(self, position: np.ndarray) -> MissionDecision: ...
