import numpy as np

from swarm_rescue.solutions.data_types import MissionDecision
from swarm_rescue.solutions.mapping import LidarMapper
from swarm_rescue.solutions.localisation import MeasuredState, SemanticState

ARRIVAL_THRESHOLD = 35.0 # exploration reach threshold
RESCUE_THRESHOLD = 5.0 # exploration reach threshold
MIN_CLUSTER_SIZE = 50
MIN_DISTANCE = 50.0

class Mission:
    def __init__(self, goal = None):
        self.goal = goal

class GlobalPlanner:
    def __init__(self):
        # PM always init in exploration mode
        self.mission = ExplorationMission()

    # PM : TODO add missions and args with new configurations
    # such as : EscortMission(Mission) - drones exploring by pair (1 Explore and 1 Escort) to identify kill zone for example
    # has_received_leader_ping: bool - get keep-alive signal from the escort leader (monitor kill-zone)

    def update_mission(self, rescue_center_position: tuple,
                       semantic_state: SemanticState):
        """Compute the current drone task regarding some informations from the drone,
         and the current mission

        Args:
            is_wounded_found (bool): _description_
            is_rescue_center_found (bool): _description_
            has_grasped_wounded (bool): _description_

        Returns:
            Mission: _description_
        """
        if isinstance(self.mission, ExplorationMission) and semantic_state.has_found_wounded and not semantic_state.has_grasped_wounded:
            self.mission = RescueMission(semantic_state.wounded_list[0])
            print(self.mission)

        elif isinstance(self.mission, RescueMission) and semantic_state.has_grasped_wounded:
            self.mission = DropMission(rescue_center_position)
            print(self.mission)

        elif isinstance(self.mission, DropMission) and not semantic_state.has_grasped_wounded:
            self.mission = ExplorationMission()
            print(self.mission)

        # PM TODO : Update with further mission types


        return True

    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        """Call child method, depending on the affected mission

        Args:
            position (_type_): _description_
            mapper (LidarMapper): _description_

        Returns:
            MissionDecision: _description_
        """
        return self.mission.decide(estimator, semantic_state, mapper)

class RescueMission(Mission):
    """Initialized with a wounded position."""
    def __init__(self, goal = None):
        self.goal = goal 

    def update_goal(self, semantic_state:SemanticState) -> tuple:
        """Update goal for a moving wounded

        Args:
            semantic_state (SemanticState): _description_

        Returns:
            tuple: _description_
        """
        if(len(semantic_state.wounded_list) > 0):
            return semantic_state.wounded_list[0]
        else:
            return self.goal
            
    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        """Always set grasper to 1 when rescuing

        Args:
            position (tuple): _description_
            mapper (LidarMapper): _description_

        Returns:
            MissionDecision: _description_
        """
        self.goal = self.update_goal(semantic_state)
        
        return MissionDecision(self.goal, 1)
    
class DropMission(Mission):
    """Initialized with the rescue center position."""
    def __init__(self, goal = None):
        self.goal = goal
        self.exploration_path = [(20, -180),(-50, -180),(-50, 170),(-310, 170),(-310, -180),]
        self.index = len(self.exploration_path) - 1

    def update_goal(self, position, mapper:LidarMapper) -> tuple:
        if self.index > 0:
            self.index -= 1

        return self.exploration_path[self.index]
            # return mapper.select_exploration_waypoint(
            #         position,
            #         minimum_cluster_size = MIN_CLUSTER_SIZE,
            #         minimum_distance = MIN_DISTANCE,
            #         )
    
    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        if estimator.position_ is None:
            return MissionDecision(None, 0)

        if self.goal is None:
            self.goal = self.update_goal(estimator.position_, mapper)

        goal_reached = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < ARRIVAL_THRESHOLD
        )

        if goal_reached:
            self.goal = self.update_goal(estimator.position_, mapper)

        return MissionDecision(self.goal, 0)

class ExplorationMission(Mission):
    """Refresh exploration goals when frontiers change or the goal is reached."""
    def __init__(self):
        self.goal = None
        self.exploration_path = [(20, -180),(-50, -180),(-50, 170),(-310, 170),(-310, -180),]
        self.index = -1

    def update_goal(self, position, mapper:LidarMapper) -> tuple:
        if self.index < len(self.exploration_path) - 1:
            self.index += 1

        return self.exploration_path[self.index]
            # return mapper.select_exploration_waypoint(
            #         position,
            #         minimum_cluster_size = MIN_CLUSTER_SIZE,
            #         minimum_distance = MIN_DISTANCE,
            #         )
    
    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        if estimator.position_ is None:
            return MissionDecision(None, 0)

        if self.goal is None:
            self.goal = self.update_goal(estimator.position_, mapper)

        goal_reached = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < ARRIVAL_THRESHOLD
        )

        if goal_reached:
            self.goal = self.update_goal(estimator.position_, mapper)

        return MissionDecision(self.goal, 0)