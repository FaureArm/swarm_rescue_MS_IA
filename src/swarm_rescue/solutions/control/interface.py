from typing import Protocol

from swarm_rescue.simulation.drone.controller import CommandsDict

from ..data_types import MissionIntent, NavigationResult, StateEstimate


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
