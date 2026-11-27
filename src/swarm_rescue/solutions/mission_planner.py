import numpy as np

from swarm_rescue.solutions.data_types import MissionDecision, Activity, Waypoint, MotionReference
from swarm_rescue.solutions.mapper import LidarMapper
from swarm_rescue.solutions.path_follower import PathFollower
from swarm_rescue.solutions.explorer import Explorer
from swarm_rescue.solutions.localisation import MeasuredState, SemanticState

EXPLORATION_THRESHOLD = 35.0 # exploration threshold
APPROACHING_THRESHOLD = 100.0 # exploration threshold
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
            self.mission = RescueMission(semantic_state.wounded_position_list[0])

        elif isinstance(self.mission, RescueMission) and has_graspered_wounded:
            self.mission = DropMission(rescue_center_position)

        elif isinstance(self.mission, DropMission) and not has_graspered_wounded:
            self.mission = StandBy()

        return True

    def decide(self, 
               estimator:MeasuredState, 
               semantic_state:SemanticState, 
               mapper:LidarMapper,
               explorer:Explorer,
               path_follower:PathFollower) -> MissionDecision:
        """Call child method, depending on the affected mission
        """
        
        return self.mission.decide(estimator, 
                                   semantic_state, 
                                   mapper,
                                   explorer,
                                   path_follower)



class ExplorationMission(Mission):
    """Refresh exploration goals when frontiers change or the goal is reached."""
    def __init__(self):
        self.goal = None
        self.motion_reference = None
        # # PM TODO: to remove once the exploration system is plugged
        # self.exploration_path = [(20, -180),(-50, -180),(-50, 170),(-310, 170),(-310, -180),]
        # self.index = -1

    def update_goal(self, 
                    estimator:MeasuredState, 
                    path_follower:PathFollower) -> MotionReference:
        
        return path_follower.update(estimator)

    
    def decide(self, 
               estimator:MeasuredState, 
               semantic_state:SemanticState, # useless here
               mapper:LidarMapper,
               explorer:Explorer,
               path_follower:PathFollower) -> MissionDecision:

        if estimator.position_ is None or path_follower is None:
            return MissionDecision(None, 0, Activity.EXPLORING)

        need_waypoint_update = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < EXPLORATION_THRESHOLD
        )

        if self.goal is None or need_waypoint_update:
            # PM TODO : explorer entry point - switch mapper to explorer
            next_point = explorer.select_exploration_waypoint(estimator.position_)

            if next_point is not None:
                path_follower.set_path(path=[Waypoint(next_point[0], next_point[1], 10, None),],
                                        lookahead=80.0,
                                        gain=0.05,
                                        max_pos=5.0,
                                        max_vel=10,)

            self.motion_reference = self.update_goal(estimator, path_follower)

            self.goal = self.motion_reference.target

        return MissionDecision(self.motion_reference, 0, Activity.EXPLORING)


    
class RescueMission(Mission):
    """Initialized with a wounded position."""
    def __init__(self, goal = None):
        self.goal = goal 

    def update_goal(self, 
                    semantic_state:SemanticState) -> MotionReference:
        """Update goal for a moving wounded
        """
        
        if(len(semantic_state.wounded_position_list) > 0):
            self.goal = semantic_state.wounded_position_list[-1]

        # PM TODO : set speed to None when approaching wounded
        # to use the previous PD controller method
        motion_reference = MotionReference(self.goal, None, None)
        
        return motion_reference
            
    def decide(self, 
               estimator:MeasuredState, # useless here
               semantic_state:SemanticState, 
               mapper:LidarMapper,
               explorer:Explorer,
               path_follower:PathFollower # useless here
               ) -> MissionDecision:
        """Always set grasper to 1 when rescuing
        """
        motion_reference = self.update_goal(semantic_state)
            
        return MissionDecision(motion_reference, 1, Activity.GRASPING_WOUNDED)




    
class DropMission(Mission):
    """Initialized with the rescue center position."""
    def __init__(self, goal = None):
        self.goal = goal
        self.is_first_mission = True

    def update_goal(self, estimator:MeasuredState, mapper:LidarMapper, explorer:Explorer, path_follower:PathFollower) -> tuple:
        """_summary_

        Args:
            position (_type_): _description_
            mapper (LidarMapper): _description_

        Returns:
            tuple: _description_
        """
        # PM TODO: to remove once the navigation system is plugged
        
        # check if a path has been given
        if not path_follower.has_path:
            # follow the past journey in reverse mode 
            # to come back to the spawn point
            path = []
            for waypoint in explorer.journey[::-1]:
                path.append(Waypoint(waypoint[0], waypoint[1], 20, None))

                path_follower.set_path(path=path,
                                    lookahead=80.0,
                                    gain=0.05,
                                    max_pos=5.0,
                                    max_vel=10,)
        
        return path_follower.update(estimator)
    
    
    def decide(self, 
               estimator:MeasuredState,
               semantic_state:SemanticState, # useless here
               mapper:LidarMapper,
               explorer:Explorer,
               path_follower:PathFollower
               ) -> MissionDecision:
        """_summary_

        Args:
            estimator (MeasuredState): _description_
            semantic_state (SemanticState): _description_
            mapper (LidarMapper): _description_

        Returns:
            MissionDecision: _description_
        """
        # PM TODO : trick to set a new path
        # there must be a better way
        if self.is_first_mission:
            path_follower.has_path = False
            self.is_first_mission = False

        if estimator.position_ is None:
            return MissionDecision(None, 1, Activity.SEARCHING_RESCUE_CENTER)
    
        motion_reference = self.update_goal(estimator, mapper, explorer, path_follower)

        # PM: different reaching thresholds between waypoints and rescue center
        approaching_rescue_center = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < APPROACHING_THRESHOLD
        )

        rescue_center_reached = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < EXPLORATION_THRESHOLD
        )

        # first contact with rescue center
        if approaching_rescue_center:
            return MissionDecision(motion_reference, 1, Activity.DROPPING_AT_RESCUE_CENTER)

        elif rescue_center_reached:
            return MissionDecision(motion_reference, 0, Activity.DROPPING_AT_RESCUE_CENTER)
        
        return MissionDecision(motion_reference, 1, Activity.SEARCHING_RESCUE_CENTER)



        
class StandBy(Mission):
    """Stop the drone at its last position."""

    def decide(self, 
               estimator:MeasuredState,
               semantic_state:SemanticState, # useless here
               mapper:LidarMapper,
               path_follower:PathFollower
               ) -> MissionDecision:
        """Always set grasper to 1 when rescuing
        """
        return MissionDecision(MotionReference(estimator.position_, None, None), 0, Activity.STAND_BY)

