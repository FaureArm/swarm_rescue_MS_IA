from swarm_rescue.solutions.control import PDController
from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.localisation import MeasuredState, SemanticState
from swarm_rescue.solutions.mapper import LidarMapper
from swarm_rescue.solutions.mission_planner import GlobalPlanner
from swarm_rescue.solutions.explorer import Explorer
from swarm_rescue.solutions.path_follower import PathFollower

# NB: constants for testing, might want to move these elsewhere 
# for cleanless

### Map exploration constants ###
FRONTIER_UPDATE_INTERVAL = 100 # whatever happens, frontier is updated every x ticks; which causes recalculation of exploration waypoint as well
ARRIVAL_THRESHOLD = 50 # distance threshold to frontier exploration waypoint
MINIMUM_CLUSTER_SIZE = 50 # minimum cluster size to define a frontier
MINIMUM_DISTANCE = MINIMUM_CLUSTER_SIZE + 0 # minimum distance to drone for our waypoint candidates for frontier exploration. NB needs to be bigger than arrival threshold

def drone_components(*, heading_goal=None):
    mapper = LidarMapper(frontier_update_interval=FRONTIER_UPDATE_INTERVAL)
    return {
        "planner": GlobalPlanner(),
        "estimator": MeasuredState(),
        "semantic_state": SemanticState(),
        "controller": PDController(heading_goal=heading_goal),
        "mapper": mapper,
        "explorer": Explorer(mapper),
        "path_follower":PathFollower()
    }

class MyDroneEval(RescueDrone):
    def __init__(self, **kwargs):
        super().__init__(**drone_components(), **kwargs)
        # super().__init__(**kwargs)
