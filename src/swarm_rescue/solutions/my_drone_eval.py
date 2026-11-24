from swarm_rescue.solutions.control import PDController
from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.localisation import MeasuredState
from swarm_rescue.solutions.mapping import LidarMapper
from swarm_rescue.solutions.mission_planner import (
    FrontierExplorationMission,
    WaypointMission,
)
from swarm_rescue.solutions.data_types import Waypoint
from swarm_rescue.solutions.path_follower import PathFollower

# NB: constants for testing, might want to move these elsewhere 
# for cleanless

### Map exploration constants ###
FRONTIER_UPDATE_INTERVAL = 100 # whatever happens, frontier is updated every x ticks; which causes recalculation of exploration waypoint as well
ARRIVAL_THRESHOLD = 50 # distance threshold to frontier exploration waypoint
MINIMUM_CLUSTER_SIZE = 50 # minimum cluster size to define a frontier
MINIMUM_DISTANCE = MINIMUM_CLUSTER_SIZE + 0 # minimum distance to drone for our waypoint candidates for frontier exploration. NB needs to be bigger than arrival threshold


def waypoint_components(*, mapping=False, goal=None, heading_goal=None):
    # initialise components for waypoint flight, with optional mapping
    return {
        "estimator": MeasuredState(),
        "mission": WaypointMission(goal=goal, arrival_threshold=1 if mapping else 0.5),
        "controller": PDController(heading_goal=heading_goal),
        "mapper": LidarMapper() if mapping else None,
    }


def exploration_components(*, heading_goal=None):
    mapper = LidarMapper( frontier_update_interval=FRONTIER_UPDATE_INTERVAL)
    return {
        "estimator": MeasuredState(),
        "mission": FrontierExplorationMission(mapper = mapper, 
                                              arrival_threshold = ARRIVAL_THRESHOLD, 
                                              minimum_cluster_size = MINIMUM_CLUSTER_SIZE, 
                                              minimum_distance = MINIMUM_DISTANCE),
        "controller": PDController(heading_goal=heading_goal),
        "mapper": mapper,
    }

def path_components():
    path = [
        Waypoint(295.0, 118.0, 20, None),
        Waypoint(170.0, -165.0, 15, None),
        Waypoint(-60.0, -165.0, 10, None),
        Waypoint(-100.0, -125.0, 10, None),
        Waypoint(-100.0, 125.0, 15, None),
        Waypoint(-140.0, 165.0, 15, None),
        Waypoint(-280.0, 165.0, 10, None),
        Waypoint(-320.0, 125.0, 5, None),
        Waypoint(-320.0, -110.0, 0.0, None),
    ]

    return {
        "estimator": MeasuredState(),
        "mission": WaypointMission(
            goal=(path[-1].x, path[-1].y),
            arrival_threshold=0,
        ),
        "controller": PDController(),
        "mapper": None,
        "path_follower": PathFollower(
            path=path,
            lookahead=80.0,
            gain=0.05,
            max_pos=5.0,
            max_vel=10,
        ),
    }


class MyDroneEval(RescueDrone):
    def __init__(self, **kwargs):
        super().__init__(**path_components(), **kwargs)
