import numpy as np

from ..data_types import Waypoint
from ..world.grid import OccupancyGrid
from .traversability import segment_is_traversable


class WaypointProfile:
    def __init__(self, *, min_speed: float, max_speed: float):
        self.min_speed_ = min_speed
        self.max_speed_ = max_speed

    def prepare(self, route: np.ndarray, grid: OccupancyGrid | None = None,
                mask: np.ndarray | None = None, *, step: int = 10) -> list[Waypoint]:
        """ Create a path of waypoints from a route"""
        points = np.asarray(route, dtype=float)
        if len(points) <= 1:
            return [Waypoint(*point, 0.0, None) for point in points]

        # slice the route to only keep a number of points
        sampled = points[::step]

        # make sure we keep the final point/goal
        # in case the slice skipped it
        if (len(points) - 1) % step:
            sampled = np.vstack((sampled, points[-1]))

        # If sampling cuts a blocked corner, keep the original path geometry.
        if grid is not None and mask is not None and any(
            not segment_is_traversable(grid, mask, start, end)
            for start, end in zip(sampled, sampled[1:])
        ):
            sampled = points

        # estimate speed for each of these points
        # and transform them into waypoints
        # TODO: might want to assign a heading? not sure
        sampled, speeds = self.calculate_speeds(sampled)
        return [Waypoint(x, y, speed, None) for (x, y), speed in zip(sampled, speeds)]



    def calculate_speeds(self, route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
