from typing import Protocol

import numpy as np

from ..data_types import (
    MissionIntent, MotionReference, NavigationResult, Observations, StateEstimate, Waypoint, WorldView,
)
from ..world.grid import OccupancyGrid


class NavigationContext(Protocol):
    @property
    def pose(self) -> StateEstimate | None: ...

    @property
    def world(self) -> WorldView: ...

    @property
    def tick_observations(self) -> Observations: ...

    @property
    def intent(self) -> MissionIntent: ...


class Navigation(Protocol):
    def update(self, drone: NavigationContext) -> NavigationResult:
        """Handle motion requests and return the reference, status and debug data."""
        ...


class PathPlanner(Protocol):
    def plan_path(
        self, grid: OccupancyGrid, start: np.ndarray, goal: np.ndarray,
        *, traversable: np.ndarray,
    ) -> np.ndarray | None:
        """Return ordered world-xy points from start to goal, or None.

        The borrowed boolean mask uses grid [x, y] indices and defines allowed
        cells for this attempt. Treat the mask and grid as read-only.
        """
        ...


class Tracking(Protocol):
    def load(self, waypoints: list[Waypoint]) -> None:
        """Replace the route and reset tracking progress."""
        ...

    def track(self, pose: StateEstimate) -> MotionReference:
        """Advance the loaded route using the current pose."""
        ...
