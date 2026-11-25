import numpy as np

from swarm_rescue.solutions.data_types import MissionDecision, Activity
from swarm_rescue.solutions.mapping import LidarMapper
from swarm_rescue.solutions.localisation import MeasuredState, SemanticState

EXPLORATION_THRESHOLD = 35.0 # exploration threshold
RESCUE_THRESHOLD = 10.0 # rescue threshold
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

    def update_mission(self, 
                       rescue_center_position: tuple,
                       semantic_state: SemanticState,
                       has_graspered_wounded: bool):
        """Compute the current drone task regarding some informations from the drone,
         and the current mission
        """
        # PM TODO : Update with further mission types
        if isinstance(self.mission, ExplorationMission) and semantic_state.has_found_wounded and not has_graspered_wounded:
            self.mission = RescueMission(semantic_state.wounded_list[0])

        elif isinstance(self.mission, RescueMission) and has_graspered_wounded:
            self.mission = DropMission(rescue_center_position)

        elif isinstance(self.mission, DropMission) and not has_graspered_wounded:
            print("back to exploration")
            self.mission = ExplorationMission()

        return True

    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        """Call child method, depending on the affected mission
        """
        return self.mission.decide(estimator, semantic_state, mapper)





class ExplorationMission(Mission):
    """Refresh exploration goals when frontiers change or the goal is reached."""
    def __init__(self):
        self.goal = None
        # PM TODO: to remove once the exploration system is plugged
        self.exploration_path = [(20, -180),(-50, -180),(-50, 170),(-310, 170),(-310, -180),]
        self.index = -1

    def update_goal(self, position, mapper:LidarMapper) -> tuple:
        # PM TODO: to remove once the exploration system is plugged
        if self.index < len(self.exploration_path) - 1:
            self.index += 1

        return self.exploration_path[self.index]
        # PM TODO: to uncomment once the exploration system is plugged
            # return mapper.select_exploration_waypoint(
            #         position,
            #         minimum_cluster_size = MIN_CLUSTER_SIZE,
            #         minimum_distance = MIN_DISTANCE,
            #         )
    
    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        if estimator.position_ is None:
            return MissionDecision(None, 0, Activity.EXPLORING)

        if self.goal is None:
            self.goal = self.update_goal(estimator.position_, mapper)

        goal_reached = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < EXPLORATION_THRESHOLD
        )

        if goal_reached:
            self.goal = self.update_goal(estimator.position_, mapper)

        return MissionDecision(self.goal, 0, Activity.EXPLORING)




    
class RescueMission(Mission):
    """Initialized with a wounded position."""
    def __init__(self, goal = None):
        self.goal = goal 

    def update_goal(self, semantic_state:SemanticState) -> tuple:
        """Update goal for a moving wounded
        """
        if(len(semantic_state.wounded_list) > 0):
            return semantic_state.wounded_list[-1]
        else:
            return self.goal
            
    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        """Always set grasper to 1 when rescuing
        """
        self.goal = self.update_goal(semantic_state)
            
        return MissionDecision(self.goal, 1, Activity.GRASPING_WOUNDED)




    
class DropMission(Mission):
    """Initialized with the rescue center position."""
    def __init__(self, goal = None):
        self.goal = goal
        # PM TODO: to remove once the navigation system is plugged
        self.exploration_path = [(20, -180),(-50, -180),(-50, 170),(-310, 170),(-310, -180),]
        self.index = len(self.exploration_path) -1
        self.intermediate_goal = self.exploration_path[-1]

    def update_goal(self, position, mapper:LidarMapper) -> tuple:
        """_summary_

        Args:
            position (_type_): _description_
            mapper (LidarMapper): _description_

        Returns:
            tuple: _description_
        """
        # PM TODO: to remove once the navigation system is plugged
        if self.index > 0:
            self.index -= 1

            return self.exploration_path[self.index]
        
        else:
            return self.goal

        # PM TODO: to uncomment once the navigation system is plugged
            # return mapper.select_exploration_waypoint(
            #         position,
            #         minimum_cluster_size = MIN_CLUSTER_SIZE,
            #         minimum_distance = MIN_DISTANCE,
            #         )
    
    def decide(self, estimator:MeasuredState, semantic_state:SemanticState, mapper:LidarMapper) -> MissionDecision:
        """_summary_

        Args:
            estimator (MeasuredState): _description_
            semantic_state (SemanticState): _description_
            mapper (LidarMapper): _description_

        Returns:
            MissionDecision: _description_
        """
        if estimator.position_ is None:
            return MissionDecision(None, 1, Activity.SEARCHING_RESCUE_CENTER)
    
        # PM: different reaching thresholds between waypoints and rescue center
        rescue_center_reached = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < EXPLORATION_THRESHOLD
        )

        intermediate_goal_reached = (
            self.intermediate_goal is not None
            and np.linalg.norm(self.intermediate_goal - estimator.position_) < EXPLORATION_THRESHOLD
        )

        if rescue_center_reached:
            return MissionDecision(self.goal, 0, Activity.DROPPING_AT_RESCUE_CENTER)

        elif intermediate_goal_reached:
            self.intermediate_goal = self.update_goal(estimator.position_, mapper)

        return MissionDecision(self.intermediate_goal, 1, Activity.SEARCHING_RESCUE_CENTER)
        
