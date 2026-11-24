import numpy as np

from swarm_rescue.solutions.contracts.goals import MissionDecision
from swarm_rescue.solutions.decision.exploration.interface import ExplorationPolicy


class FrontierExplorationMission:
    """Refresh exploration goals when frontiers change or the goal is reached."""
    def __init__(
        self, *, exploration: ExplorationPolicy, arrival_threshold=5.0,
        minimum_cluster_size=50, minimum_distance=50.0,
    ):
        self.exploration = exploration
        self.goal = None
        self.arrival_threshold = arrival_threshold
        self.minimum_cluster_size = minimum_cluster_size
        self.minimum_distance = minimum_distance
        self.role = "exploration"
        self._last_frontier_revision = 0

    def decide(self, position) -> MissionDecision:
        if position is None:
            return MissionDecision(None, 0)

        frontier_revision = self.exploration.frontier_revision
        frontiers_updated = frontier_revision != self._last_frontier_revision

        goal_reached = (
            self.goal is not None
            and np.linalg.norm(self.goal - position) < self.arrival_threshold
        )

        if frontiers_updated or goal_reached:
            self.goal = self.exploration.select_exploration_waypoint(
                position,
                minimum_cluster_size=self.minimum_cluster_size,
                minimum_distance=self.minimum_distance,
            )

        self._last_frontier_revision = frontier_revision
        return MissionDecision(self.goal, 0)
