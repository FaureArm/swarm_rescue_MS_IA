from typing import Protocol, TypedDict

from swarm_rescue.solutions.config import (
    ARRIVAL_THRESHOLD,
    FRONTIER_UPDATE_INTERVAL,
    MINIMUM_CLUSTER_SIZE,
    MINIMUM_DISTANCE,
)
from swarm_rescue.solutions.control.interface import MotionController
from swarm_rescue.solutions.control.pd_controller import PDController
from swarm_rescue.solutions.decision.exploration.exploration_behavior import FrontierExplorationMission
from swarm_rescue.solutions.decision.exploration.frontier_detector import FrontierDetector
from swarm_rescue.solutions.decision.exploration.interface import ExplorationPolicy
from swarm_rescue.solutions.decision.exploration.nearest_frontier_policy import NearestFrontierPolicy
from swarm_rescue.solutions.decision.interface import Behavior
from swarm_rescue.solutions.decision.waypoint_behavior import WaypointMission
from swarm_rescue.solutions.estimation.interface import StateEstimator
from swarm_rescue.solutions.estimation.measured_state import MeasuredState
from swarm_rescue.solutions.mapping.interface import Mapper
from swarm_rescue.solutions.mapping.lidar_mapper import LidarMapper
from swarm_rescue.solutions.navigation.direct_navigator import DirectNavigator
from swarm_rescue.solutions.navigation.interface import Navigator


class DroneComponents(TypedDict):
    estimator: StateEstimator
    mission: Behavior
    controller: MotionController
    mapper: Mapper | None
    navigator: Navigator
    exploration: ExplorationPolicy | None


class ComponentFactory(Protocol):
    """Build fresh components after the adapter knows the map dimensions."""

    def __call__(self, *, map_size: tuple[int, int] | None = None) -> DroneComponents: ...


def waypoint_components(
    *, mapping=False, goal=None, heading_goal=None, map_size=None,
) -> DroneComponents:
    # initialise components for waypoint flight, with optional mapping
    mapper = LidarMapper(map_size=map_size if map_size is not None else (2000, 2000)) if mapping else None
    exploration = NearestFrontierPolicy(FrontierDetector(mapper)) if mapper is not None else None
    return {
        "estimator": MeasuredState(),
        "mission": WaypointMission(goal=goal, arrival_threshold=1 if mapping else 0.5),
        "controller": PDController(heading_goal=heading_goal),
        "mapper": mapper,
        "navigator": DirectNavigator(),
        "exploration": exploration,
    }


def exploration_components(*, heading_goal=None, map_size=None) -> DroneComponents:
    mapper = LidarMapper(map_size=map_size if map_size is not None else (2000, 2000))
    exploration = NearestFrontierPolicy(
        FrontierDetector(mapper, frontier_update_interval=FRONTIER_UPDATE_INTERVAL)
    )
    return {
        "estimator": MeasuredState(),
        "mission": FrontierExplorationMission(exploration = exploration, 
                                              arrival_threshold = ARRIVAL_THRESHOLD, 
                                              minimum_cluster_size = MINIMUM_CLUSTER_SIZE, 
                                              minimum_distance = MINIMUM_DISTANCE),
        "controller": PDController(heading_goal=heading_goal),
        "mapper": mapper,
        "navigator": DirectNavigator(),
        "exploration": exploration,
    }
