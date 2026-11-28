from typing import Protocol

import numpy as np

from ..data_types import Detection, Observations, StateEstimate, WorldView
from .grid import OccupancyGrid


class WorldContext(Protocol):
    @property
    def tick_observations(self) -> Observations: ...

    @property
    def pose(self) -> StateEstimate | None: ...


class Mapping(Protocol):
    @property
    def grid(self) -> OccupancyGrid: ...

    @property
    def frontiers(self) -> np.ndarray | None: ...

    @property
    def frontier_revision(self) -> int: ...

    def integrate(
        self, pose: StateEstimate, ranges: np.ndarray | None,
        angles: np.ndarray | None, timestep: int,
    ) -> None: ...


class ObjectTracking(Protocol):
    def update(
        self, pose: StateEstimate, detections: tuple[Detection, ...],
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray | None]: ...


class World(Protocol):
    def update(self, drone: WorldContext) -> WorldView:
        """Update the map and publish the current view. Retain map on missing pose."""
        ...
