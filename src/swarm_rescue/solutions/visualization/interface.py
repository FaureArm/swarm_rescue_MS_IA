from typing import Protocol

from ..data_types import MissionIntent, NavigationResult, Observations, StateEstimate, WorldView

"""Define what this subsystem can read and what it must provide.

These protocols document the required methods, inputs, and return values.
Any implementation that follows these contracts can be swapped in through
assembly.py without changing the code that calls it.
Implementations must provide the methods defined here and return the specified values.
"""

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
