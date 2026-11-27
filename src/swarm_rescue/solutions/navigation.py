import numpy as np
from skimage.morphology import isotropic_dilation
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from swarm_rescue.solutions.data_types import OccupancyGrid, Waypoint

# this function remains the same for everyone, so
# we keep it outside of path algo for now
def build_traversable_mask(
    grid: OccupancyGrid, *, clearance_px: float = 15.0,
) -> np.ndarray:
    """Return a boolean [x, y] mask of free cells with obstacle clearance."""
    
    if not np.isfinite(clearance_px) or clearance_px < 0:
        raise ValueError("clearance_px must be finite and nonnegative")

    states = grid.classify_cells()
    occupied = states == grid.OCCUPIED

    # add the clearance to make sure the drone doesnt hit walls
    padded_obstacles = np.pad(occupied, 1, constant_values=True)
    # we use an already existing library for this, makes things simpler
    inflated_obstacles = isotropic_dilation(padded_obstacles, radius=clearance_px)
    clearance_zone = ~inflated_obstacles[1:-1, 1:-1] #remove padding and flip booleans

    # return cells that are free and touside clearance zone
    return (states == grid.FREE) & clearance_zone


class AstartPlanner():
    def __init__(self, *, clearance_px = 25.0, max_speed = 20, min_speed = 1):
        self.clearance_px = clearance_px
        self.current_route = None
        self.max_speed_ = max_speed
        self.min_speed_ = min_speed


    def get_traversable_mask(self, grid: OccupancyGrid) -> np.ndarray:
        return build_traversable_mask(grid, clearance_px=self.clearance_px)

    def plan_path(self, grid: OccupancyGrid, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        """Return world-coordinate waypoints, or None if no route exists"""
        #TODO: Perhaps could try to adapt to accepts waypoints directly
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float) # 'final target'
       
        if start.shape != (2,) or goal.shape != (2,):
            raise ValueError("start and goal must be (x, y) points")

        # get our coords in the coord of our grid
        world_frame_start_x, world_frame_start_y, start_valid = grid.world_to_grid(*start)
        world_frame_goal_x, world_frame_goal_y, goal_valid = grid.world_to_grid(*goal)
        if not (start_valid and goal_valid):
            return None

        # get all the available spots/cells a route can go through
        mask = self.get_traversable_mask(grid)

        ## Calculate path ##
        # using an imported library
        start_cell = (int(world_frame_start_x), int(world_frame_start_y))
        goal_cell = (int(world_frame_goal_x), int(world_frame_goal_y))
        if not (mask[start_cell] and mask[goal_cell]):
            return None

        if start_cell == goal_cell:
            if np.array_equal(start, goal):
                return start[None, :]
            return np.stack((start, goal))

        # NB: our mask uses [x, y]but the imported library reads 
        # its matrix as [y, x], so we need to switch
        search_grid = Grid(matrix=mask.T)
        finder = AStarFinder(diagonal_movement=DiagonalMovement.only_when_no_obstacle)
        nodes, _ = finder.find_path(
            search_grid.node(*start_cell),
            search_grid.node(*goal_cell),
            search_grid,
        )
        if not nodes:
            return None

        # retranslate back to our grid coords system
        # we calculte the same origin as we do in the occupancygrid datatype
        origin = np.array([grid.width // 2, grid.height // 2])
        # here we recalculate from our 'origin'
        route = np.array([(node.x, node.y) for node in nodes], dtype=float) - origin
        # world to grid func rounds values, so we replace actually known values at least
        route[0], route[-1] = start, goal
        return route

    def extract_waypoints(self, route: np.ndarray, step = 10) -> list[Waypoint]:
        """ Create a path of waypoints from a route"""
        points = np.asarray(route, dtype=float)
        if len(points) <= 1:
            return points.copy()

        # slice the route to only keep a number of points
        sampled = points[::step]

        # make sure we keep the final point/goal
        # in case the slice skipped it
        if (len(points) - 1) % step:
            sampled = np.vstack((sampled, points[-1]))

        # estimate speed for each of these points
        # and transform them into waypoints
        # TODO: might want to assign a heading? not sure
        sampled, speeds = self.calcute_speeds(sampled)
        return [Waypoint(x, y, speed, None) for (x, y), speed in zip(sampled, speeds)]



    def calcute_speeds(self, route: np.ndarray) -> np.ndarray:
        """ Estimate the best speed from a series of points based on the curvature"""
        # we initialise our array with a default speed val, then adjust
        speeds = np.full(len(route), self.min_speed_, dtype=float)

        # adjust speed based on angle
        for i in range(1, len(route) - 1):
            incoming = route[i] - route[i - 1]
            outgoing = route[i + 1] - route[i]
            cos_angle = np.dot(incoming, outgoing) / (np.linalg.norm(incoming) * np.linalg.norm(outgoing))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            speeds[i] = max(self.min_speed_, self.max_speed_ / (1 + 3 * angle))
        #print(speeds)

        # set our final target speed at 0
        #TODO: could perhaps be a problem?
        if len(speeds):
            speeds[-1] = 0.0

        # we apply a backwards braking pass
        # the first init of speed can be too abrupt in changes
        # leading to worse performances/overshoot bc drone doesnt
        # have time to slow down. With this we apply breaking earlier
        braking = 0.5
        for i in range(len(route) - 2, -1, -1):
            distance = np.linalg.norm(route[i + 1] - route[i])
            # based on distance to next waypoint and on our breaking capacity,
            # we try to lower the speed if necessary based on next waypoint speed
            speed_limit = np.sqrt(speeds[i + 1] ** 2 + 2 * braking * distance)
            speeds[i] = min(speeds[i], speed_limit)

        

        return route, speeds



