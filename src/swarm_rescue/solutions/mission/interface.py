from typing import Protocol

import numpy as np

from ..data_types import MissionIntent, NavigationResult, Observations, StateEstimate, WorldView

"""Define what this subsystem can read and what it must provide.

These protocols document the required methods, inputs, and return values.
Any implementation that follows these contracts can be swapped in through
assembly.py without changing the code that calls it.
Implementations must provide the methods defined here and return the specified values.
"""

class MissionContext(Protocol):
    @property
    def pose(self) -> StateEstimate | None: ...

    @property
    def world(self) -> WorldView: ...

    @property
    def tick_observations(self) -> Observations: ...

    @property
    def navigation_result(self) -> NavigationResult:
        """Most recent navigation feedback, normally from the previous tick."""
        ...


class Mission(Protocol):
    def update(self, drone: MissionContext) -> MissionIntent:
        """Choose destination and grasper intent without starting a route."""
        ...


class ExplorationPolicy(Protocol):
    def candidates(self, world: WorldView, drone_position: np.ndarray) -> np.ndarray:
        """Return an (N, 2) array of candidate goals in world pixels."""
        ...
