from typing import Protocol

from swarm_rescue.simulation.drone.controller import CommandsDict

from ..data_types import MissionIntent, NavigationResult, StateEstimate

"""Define what this subsystem can read and what it must provide.

These protocols document the required methods, inputs, and return values.
Any implementation that follows these contracts can be swapped in through
assembly.py without changing the code that calls it.
Implementations must provide the methods defined here and return the specified values.
"""

class ControlContext(Protocol):
    @property
    def pose(self) -> StateEstimate | None: ...

    @property
    def intent(self) -> MissionIntent: ...

    @property
    def navigation_result(self) -> NavigationResult: ...


class Control(Protocol):
    def update(self, drone: ControlContext) -> CommandsDict:
        """Apply movement and grasper intent, including when motion is unavailable."""
        ...
