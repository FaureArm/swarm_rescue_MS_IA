from typing import Protocol

import numpy as np


class ExplorationPolicy(Protocol):
    """Update after mapping, even when localization is unavailable.

    Cached arrays are borrowed read-only views. frontier_revision increases
    on each detection refresh, including after grid replacement.
    """

    @property
    def frontier_mask_(self) -> np.ndarray | None: ...

    @property
    def frontier_waypoints_(self) -> np.ndarray: ...

    @property
    def frontier_revision(self) -> int: ...

    def update(self, timestep: int) -> None: ...

    def select_exploration_waypoint(
        self, drone_position: np.ndarray, minimum_cluster_size: int = 5,
        minimum_distance: float = 50.0,
    ) -> np.ndarray | None: ...
