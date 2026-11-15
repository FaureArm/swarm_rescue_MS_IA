from swarm_rescue.solutions.control import PDController
from swarm_rescue.solutions.drone import RescueDrone
from swarm_rescue.solutions.localisation import MeasuredState
from swarm_rescue.solutions.mapping import LidarMapper
from swarm_rescue.solutions.mission_planner import WaypointMission


def waypoint_components(*, mapping=False, goal=None, heading_goal=None):
    # initialise components for waypoint flight, with optional mapping
    return {
        "estimator": MeasuredState(),
        "mission": WaypointMission(goal=goal, arrival_threshold=1 if mapping else 0.5),
        "controller": PDController(heading_goal=heading_goal),
        "mapper": LidarMapper() if mapping else None,
    }


class MyDroneEval(RescueDrone):
    def __init__(self, **kwargs):
        super().__init__(
            **waypoint_components(mapping=True, goal=(150.0, 100.0)), **kwargs
        )
