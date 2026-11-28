import numpy as np

from ..world.grid import OccupancyGrid
from .traversability import segment_is_traversable


class StraightLinePlanner:
    """Demo planner. Return a straight segment through traversable space."""

    def plan_path(
        self, grid: OccupancyGrid, start: np.ndarray, goal: np.ndarray,
        *, traversable: np.ndarray,
    ) -> np.ndarray | None:
        if not segment_is_traversable(grid, traversable, start, goal):
            return None
        return np.stack((start, goal))
