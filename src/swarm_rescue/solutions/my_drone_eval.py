from swarm_rescue.solutions.assembly import exploration_components, waypoint_components
from swarm_rescue.solutions.config import (
    ARRIVAL_THRESHOLD,
    FRONTIER_UPDATE_INTERVAL,
    MINIMUM_CLUSTER_SIZE,
    MINIMUM_DISTANCE,
)
from swarm_rescue.solutions.drone import RescueDrone

__all__ = [
    "MyDroneEval", "exploration_components", "waypoint_components",
    "FRONTIER_UPDATE_INTERVAL", "ARRIVAL_THRESHOLD", "MINIMUM_CLUSTER_SIZE", "MINIMUM_DISTANCE",
]


class MyDroneEval(RescueDrone):
    def __init__(self, **kwargs):
        super().__init__(component_factory=exploration_components, **kwargs)
