from swarm_rescue.solutions.control import PDController
from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.localisation import MeasuredState
from swarm_rescue.solutions.mapping import LidarMapper
from swarm_rescue.solutions.mission_planner import (
    FrontierExplorationMission,
    WaypointMission,
)

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


class MyDroneEval(RescueDrone):
    def __init__(self, **kwargs):
        super().__init__(**exploration_components(), **kwargs)
