import numpy as np

from swarm_rescue.solutions.data_types import MissionDecision, Activity, Waypoint, MotionReference
        # PM TODO : Armand's algorithm 
from swarm_rescue.solutions.mapper import LidarMapper
from swarm_rescue.solutions.explorer import Explorer
        # PM TODO : Aristide's algorithm 
from swarm_rescue.solutions.mapper_aristide import Mapper_Aristide
from swarm_rescue.solutions.explorer_aristide import Explorer_Aristide
from swarm_rescue.solutions.path_follower import PathFollower
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

    def decide(self, estimator, semantic_state, mapper, explorer, path_follower, *, navigator):
        if isinstance(self.mission, DropMission):
            return self.mission.decide(estimator, semantic_state, mapper, navigator)
        if isinstance(self.mission, StandBy):
            return self.mission.decide(estimator, semantic_state, mapper, explorer)
        return self.mission.decide(estimator, semantic_state, mapper, explorer, path_follower)



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
        # PM TODO : Armand's mapper algorithm 
            #    mapper:LidarMapper,
            #    explorer:Explorer,
        # PM TODO : Aristide's algorithm 
                 mapper:Mapper_Aristide,
                 explorer:Explorer_Aristide,
               path_follower:PathFollower) -> MissionDecision:

        if estimator.position_ is None or path_follower is None:
            return MissionDecision(None, 0, Activity.EXPLORING)

        need_waypoint_update = (
            self.goal is not None
            and np.linalg.norm(self.goal - estimator.position_) < EXPLORATION_THRESHOLD
        )

        if self.goal is None or need_waypoint_update:
            position = estimator.position_
            candidates = explorer.update_frontier_waypoints(
                position, minimum_cluster_size=MIN_CLUSTER_SIZE,
                minimum_distance=MIN_DISTANCE,
            )
            if len(candidates) == 0:
                return MissionDecision(None, 0, Activity.EXPLORING)

            index = np.argmin(np.linalg.norm(candidates - position, axis=1))
            self.goal = candidates[index]
            explorer.journey.append(self.goal.copy())
            path_follower.set_path(
                [Waypoint(*position, 5, None), Waypoint(*self.goal, 0.0, None)],
                lookahead=15.0, gain=0.05, max_pos=5.0, max_vel=0.1,
            )
        return MissionDecision(path_follower.update(estimator), 0, Activity.EXPLORING)


        # # # PM TODO : Armand's algorithm 
        # #     next_point = explorer.select_exploration_waypoint(estimator.position_)



        # # PM TODO : Aristide's algorithm 
        #     next_point = None
        #     next_point_list = explorer.select_exploration_waypoint()
        # # PM TODO : Aristide's algorithm 
        #     if len(next_point_list) > 0:
        #         next_point = next_point_list[0]

        #     print("toto", next_point)
        #     if next_point is not None:
        #         path_follower.set_path(path=[Waypoint(next_point[0], next_point[1], 10, None),],
        #                                 lookahead=80.0,
        #                                 gain=0.05,
        #                                 max_pos=5.0,
        #                                 max_vel=10,)

        # # # # PM TODO : Armand's algorithm 
        # #     self.motion_reference = self.update_goal(estimator, path_follower)

        #     # self.goal = self.motion_reference.target

        # return MissionDecision(self.motion_reference, 0, Activity.EXPLORING)


    
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
        # PM TODO : Armand's mapper algorithm 
            #    mapper:LidarMapper,
            #    explorer:Explorer,
        # PM TODO : Aristide's algorithm 
                 mapper:Mapper_Aristide,
                 explorer:Explorer_Aristide,
               path_follower:PathFollower # useless here
               ) -> MissionDecision:
        """Always set grasper to 1 when rescuing
        """
        motion_reference = self.update_goal(semantic_state)
            
        return MissionDecision(motion_reference, 1, Activity.GRASPING_WOUNDED)




    
# class DropMission(Mission):
#     """Initialized with the rescue center position."""
#     def __init__(self, goal = None):
#         self.goal = goal
#         self.is_first_mission = True

#     def update_goal(self, estimator:MeasuredState, 
#         # PM TODO : Armand's mapper algorithm 
#                 #    mapper:LidarMapper,
#                 #    explorer:Explorer,
#         # PM TODO : Aristide's algorithm 
#                     mapper:Mapper_Aristide,
#                     explorer:Explorer_Aristide,
#                     path_follower:PathFollower) -> tuple:
#         """_summary_

#         Args:
#             position (_type_): _description_
#             mapper (LidarMapper): _description_

#         Returns:
#             tuple: _description_
#         """
#         # PM TODO: to remove once the navigation system is plugged
        
#         # check if a path has been given
#         if not path_follower.has_path:
#             # follow the past journey in reverse mode 
#             # to come back to the spawn point
#             path = []
#             for waypoint in explorer.journey[::-1]:
#                 path.append(Waypoint(waypoint[0], waypoint[1], 20, None))

#                 path_follower.set_path(path=path,
#                                     lookahead=80.0,
#                                     gain=0.05,
#                                     max_pos=5.0,
#                                     max_vel=10,)
        
#         return path_follower.update(estimator)
    
    
#     def decide(self, 
#                estimator:MeasuredState,
#                semantic_state:SemanticState, # useless here
#     # PM TODO : Armand's mapper algorithm 
#             #    mapper:LidarMapper,
#             #    explorer:Explorer,
#     # PM TODO : Aristide's algorithm 
#                 mapper:Mapper_Aristide,
#                 explorer:Explorer_Aristide,
#                path_follower:PathFollower
#                ) -> MissionDecision:
#         """_summary_

#         Args:
#             estimator (MeasuredState): _description_
#             semantic_state (SemanticState): _description_
#             mapper (LidarMapper): _description_

#         Returns:
#             MissionDecision: _description_
#         """
#         # PM TODO : trick to set a new path
#         # there must be a better way
#         if self.is_first_mission:
#             path_follower.has_path = False
#             self.is_first_mission = False

#         if estimator.position_ is None:
#             return MissionDecision(None, 1, Activity.SEARCHING_RESCUE_CENTER)
    
#         motion_reference = self.update_goal(estimator, mapper, explorer, path_follower)

#         # PM: different reaching thresholds between waypoints and rescue center
#         approaching_rescue_center = (
#             self.goal is not None
#             and np.linalg.norm(self.goal - estimator.position_) < APPROACHING_THRESHOLD
#         )

#         rescue_center_reached = (
#             self.goal is not None
#             and np.linalg.norm(self.goal - estimator.position_) < EXPLORATION_THRESHOLD
#         )

#         # first contact with rescue center
#         if approaching_rescue_center:
#             return MissionDecision(motion_reference, 1, Activity.DROPPING_AT_RESCUE_CENTER)

#         elif rescue_center_reached:
#             return MissionDecision(motion_reference, 0, Activity.DROPPING_AT_RESCUE_CENTER)
        
#         return MissionDecision(motion_reference, 1, Activity.SEARCHING_RESCUE_CENTER)


class DropMission(Mission):
    def __init__(self, goal=None):
        super().__init__(goal)
        self.route_requested = False

    def decide(self, estimator, semantic_state, mapper, navigator) -> MissionDecision:
        if self.goal is None:
            self.goal = semantic_state.rescue_center_position

        position = estimator.position_
        if position is None or self.goal is None:
            return MissionDecision(None, 1, Activity.SEARCHING_RESCUE_CENTER)

        distance = np.linalg.norm(self.goal - position)
        if distance < EXPLORATION_THRESHOLD:
            navigator.clear()
            return MissionDecision(None, 0, Activity.DROPPING_AT_RESCUE_CENTER)

        if not self.route_requested:
            approach_goal = np.array([350.0, 130.0])
            loaded = navigator.set_goal(mapper.occupancy_grid_, position, approach_goal)
            #loaded = navigator.set_goal(mapper.occupancy_grid_, position, self.goal)
            self.route_requested = True
            if not loaded:
                print("No A* route to the rescue center")
                self.route_requested = False
            

        role = Activity.SEARCHING_RESCUE_CENTER
        if distance < APPROACHING_THRESHOLD:
            role = Activity.DROPPING_AT_RESCUE_CENTER

        return MissionDecision(navigator.update(estimator), 1, role)
        
class StandBy(Mission):
    """Stop the drone at its last position."""

    def decide(self, 
               estimator:MeasuredState,
               semantic_state:SemanticState, # useless here
    # PM TODO : Armand's mapper algorithm 
            #    mapper:LidarMapper,
            #    explorer:Explorer,
    # PM TODO : Aristide's algorithm 
                mapper:Mapper_Aristide,
                explorer:Explorer_Aristide,
               ) -> MissionDecision:
        """Always set grasper to 1 when rescuing
        """
        return MissionDecision(MotionReference(estimator.position_, None, None), 0, Activity.STAND_BY)

