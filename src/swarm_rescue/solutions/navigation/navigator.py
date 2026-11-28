import numpy as np

from ..data_types import MovementMode, MotionReference, NavigationResult, NavigationStatus
from .interface import NavigationContext, PathPlanner, Tracking
from .traversability import build_traversable_mask, segment_is_traversable
from .waypoints import WaypointProfile


class Navigator:
    def __init__(
        self, planner: PathPlanner, follower: Tracking, profile: WaypointProfile, *,
        clearance_px: float, retry_ticks: int = 10, position_tolerance: float = 5.0,
        speed_tolerance: float = 0.1,
    ):
        self._planner = planner
        self._follower = follower
        self._profile = profile
        self._clearance = clearance_px
        self._retry_ticks = retry_ticks
        self._position_tolerance = position_tolerance
        self._speed_tolerance = speed_tolerance
        self._grid = None
        self._goal = None
        self._route = None
        self._mask = None
        self._last_attempt = 0

    def _clear_route(self):
        self._goal = self._route = self._mask = self._grid = None

    def update(self, drone: NavigationContext) -> NavigationResult:
        pose = drone.pose
        goal = drone.intent.destination
        mode = drone.intent.mode
        grid = drone.world.grid
        if pose is None or mode is MovementMode.IDLE or goal is None:
            self._clear_route()
            return NavigationResult()

        if mode is MovementMode.DIRECT:
            self._clear_route()
            return NavigationResult(NavigationStatus.FOLLOWING, MotionReference(goal, None, None))

        at_goal = np.linalg.norm(goal - pose.position) <= self._position_tolerance
        if at_goal and np.linalg.norm(pose.velocity) <= self._speed_tolerance:
            self._clear_route()
            return NavigationResult(NavigationStatus.ARRIVED)
        if at_goal and (self._route is None or not np.array_equal(goal, self._goal)):
            self._clear_route()
            return NavigationResult(
                NavigationStatus.FOLLOWING, MotionReference(goal, np.zeros(2), None),
            )

        tick = drone.tick_observations.timestep
        new_request = grid is not self._grid or not np.array_equal(goal, self._goal)
        retry = self._route is None and tick - self._last_attempt >= self._retry_ticks
        if new_request or retry:
            self._plan(grid, pose.position, goal, tick)
        if self._route is None:
            return NavigationResult(NavigationStatus.BLOCKED, traversable=self._mask)
        motion = self._follower.track(pose)
        return NavigationResult(NavigationStatus.FOLLOWING, motion, self._route, self._mask)

    def _plan(self, grid, start, goal, tick):
        self._grid, self._goal = grid, goal.copy()
        self._last_attempt = tick
        self._route = None
        self._mask = build_traversable_mask(grid, clearance_px=self._clearance)
        self._mask.flags.writeable = False
        route = self._planner.plan_path(grid, start, goal, traversable=self._mask)
        if route is None:
            return

        route = np.asarray(route, dtype=float)
        if (route.ndim != 2 or route.shape[1] != 2 or len(route) < 2
                or not np.isfinite(route).all()
                or not np.array_equal(route[0], start) or not np.array_equal(route[-1], goal)):
            raise ValueError("Planner must return finite world-xy points from start to goal")
        route = route[np.r_[True, np.any(np.diff(route, axis=0) != 0, axis=1)]]
        if len(route) < 2 or not all(
            segment_is_traversable(grid, self._mask, a, b) for a, b in zip(route, route[1:])
        ):
            return
        route.flags.writeable = False
        self._follower.load(self._profile.prepare(route, grid, self._mask))
        self._route = route
