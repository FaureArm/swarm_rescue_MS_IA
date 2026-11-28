import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from ..world.grid import OccupancyGrid


class AStarPlanner:
    def plan_path(
        self, grid: OccupancyGrid, start: np.ndarray, goal: np.ndarray,
        *, traversable: np.ndarray,
    ) -> np.ndarray | None:
        start_x, start_y, start_valid = grid.world_to_grid(*start)
        goal_x, goal_y, goal_valid = grid.world_to_grid(*goal)
        if not (start_valid and goal_valid):
            return None
        start_cell = (int(start_x), int(start_y))
        goal_cell = (int(goal_x), int(goal_y))
        if not (traversable[start_cell] and traversable[goal_cell]):
            return None
        if start_cell == goal_cell:
            return np.stack((start, goal))

        # Our grid uses [x, y]; pathfinding.Grid expects rows indexed by y.
        search_grid = Grid(matrix=traversable.T)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        nodes, _ = finder.find_path(
            search_grid.node(*start_cell), search_grid.node(*goal_cell), search_grid,
        )
        if not nodes:
            return None

        origin = np.array([grid.width // 2, grid.height // 2])
        route = np.array([(node.x, node.y) for node in nodes], dtype=float) - origin
        route[0], route[-1] = start, goal
        return route
