import numpy as np

from ..data_types import (
    Activity, MissionIntent, MovementMode, NavigationStatus,
)
from .interface import ExplorationPolicy, MissionContext


class MissionManager:
    def __init__(
        self, explorer: ExplorationPolicy, *, frontier_arrival_px: float,
        return_approach: tuple[float, float], release_distance_px: float,
        return_facing_distance_px: float,
    ):
        self._explorer = explorer
        self._frontier_arrival = frontier_arrival_px
        self._return_approach = np.array(return_approach)
        self._release_distance = release_distance_px
        self._return_facing_distance = return_facing_distance_px
        self._activity = Activity.EXPLORING
        self._goal = None
        self._wounded_goal = None
        self._revision = -1
        self._rejected: set[tuple[float, float]] = set()
        self._candidates = None

    def update(self, drone: MissionContext) -> MissionIntent:
        holding = drone.tick_observations.holding_wounded
        pose = drone.pose
        world = drone.world
        if pose is None:
            return MissionIntent(self._activity, grasper=int(holding))

        if self._activity is Activity.EXPLORING and world.wounded and not holding:
            self._wounded_goal = world.wounded[0]
            self._activity = Activity.GRASPING_WOUNDED
        if self._activity is Activity.GRASPING_WOUNDED and holding:
            self._activity = Activity.SEARCHING_RESCUE_CENTER
        elif self._activity is Activity.SEARCHING_RESCUE_CENTER and not holding:
            self._activity = Activity.STAND_BY

        if self._activity is Activity.EXPLORING:
            if world.frontier_revision != self._revision:
                self._revision = world.frontier_revision
                self._rejected.clear()
            if self._goal is not None:
                if drone.navigation_result.status is NavigationStatus.BLOCKED:
                    self._rejected.add(tuple(self._goal))
                    self._goal = None
                elif np.linalg.norm(self._goal - pose.position) < self._frontier_arrival:
                    self._goal = None
            if self._goal is None:
                self._candidates = self._explorer.candidates(world, pose.position)
                candidates = [point for point in self._candidates if tuple(point) not in self._rejected]
                if candidates:
                    self._goal = min(candidates, key=lambda p: np.linalg.norm(p - pose.position))
            return MissionIntent(
                self._activity, self._goal, MovementMode.PLANNED, candidates=self._candidates,
            )

        if self._activity is Activity.GRASPING_WOUNDED:
            if world.wounded:
                self._wounded_goal = world.wounded[-1]
            return MissionIntent(
                self._activity, self._wounded_goal, MovementMode.DIRECT,
                grasper=1, face_target=True,
            )

        if self._activity is Activity.SEARCHING_RESCUE_CENTER:
            center = world.rescue_center
            if center is None:
                return MissionIntent(self._activity, grasper=1)
            distance = np.linalg.norm(center - pose.position)
            if distance < self._release_distance:
                return MissionIntent(Activity.DROPPING_AT_RESCUE_CENTER)
            return MissionIntent(
                self._activity, self._return_approach, MovementMode.PLANNED,
                grasper=1, face_target=bool(distance < self._return_facing_distance),
            )

        return MissionIntent(Activity.STAND_BY)


class FixedGoalMission:
    def __init__(self, goal, mode: MovementMode = MovementMode.DIRECT):
        self._intent = MissionIntent(Activity.EXPLORING, np.asarray(goal, dtype=float), mode)

    def update(self, drone: MissionContext) -> MissionIntent:
        return self._intent
