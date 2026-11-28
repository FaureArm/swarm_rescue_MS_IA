from typing import Protocol

from ..data_types import MissionIntent, NavigationResult, Observations, StateEstimate, WorldView


class DebugContext(Protocol):
    @property
    def tick_observations(self) -> Observations: ...

    @property
    def pose(self) -> StateEstimate | None: ...

    @property
    def world(self) -> WorldView: ...

    @property
    def intent(self) -> MissionIntent: ...

    @property
    def navigation_result(self) -> NavigationResult: ...


class Visualization(Protocol):
    def update(self, drone: DebugContext) -> None: ...
