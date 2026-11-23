import numpy as np

from swarm_rescue.solutions.data_types import MissionDecision


class WaypointMission:
    def __init__(self, *, goal=None, arrival_threshold=0.5):
        # if no goal set, return to center of map
        self.goal = np.asarray((0.0,0.0), dtype=float) if goal is None else np.asarray(goal, dtype=float)
        # make sure not to share goals between instances, could cause issues? to check
        self.goal = self.goal.copy()
        self.position_ = None
        self.arrival_threshold = arrival_threshold
        self.role = "waypoint"

    def decide(self, position) -> MissionDecision:
        """hooses what the drone should do next, using its position estimate

        Args:
            position (_type_): _description_

        Returns:
            MissionDecision: _description_
        """
        self.position_ = position
        self.update_goal()
        return MissionDecision(self.goal, self.compute_grasper())

    def update_goal(self):
        """Functions which updates or not the current goal
        
        Returns:
            boolean: True if we managed to update our goal, False else
        """
        # change goals
        if self.goal is not None and self.position_ is not None :
            if np.linalg.norm(self.goal - self.position_) < self.arrival_threshold:
                self.goal = np.random.uniform(-100, 100, size=2)  # random new goal if close enough
                print(f"New goal: {self.goal}")

        return True


    def compute_grasper(self):
        return 0


class FrontierExplorationMission:
    """Refresh exploration goals when frontiers change or the goal is reached."""
    def __init__(
        self, *, mapper, arrival_threshold=5.0,
        minimum_cluster_size=50, minimum_distance=50.0,
    ):
        self.mapper = mapper
        self.goal = None
        self.arrival_threshold = arrival_threshold
        self.minimum_cluster_size = minimum_cluster_size
        self.minimum_distance = minimum_distance
        self.role = "exploration"
        self._last_frontier_mask = None # hold frontier masks for effenciency

    def decide(self, position) -> MissionDecision:
        if position is None:
            return MissionDecision(None, 0)

        frontier_mask = self.mapper.frontier_mask_
        frontiers_updated = frontier_mask is not self._last_frontier_mask

        goal_reached = (
            self.goal is not None
            and np.linalg.norm(self.goal - position) < self.arrival_threshold
        )

        if frontiers_updated or goal_reached:
            self.goal = self.mapper.select_exploration_waypoint(
                position,
                minimum_cluster_size=self.minimum_cluster_size,
                minimum_distance=self.minimum_distance,
            )

        self._last_frontier_mask = frontier_mask
        return MissionDecision(self.goal, 0)