from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.solutions.debug_draw import DebugDraw
from swarm_rescue.solutions.localisation import MeasuredState, SemanticState
from swarm_rescue.solutions.mission_planner import GlobalPlanner
from swarm_rescue.solutions.control import PDController
        # PM TODO : Armand's algorithm 
from swarm_rescue.solutions.mapper import LidarMapper
from swarm_rescue.solutions.explorer import Explorer
        # PM TODO : Aristide's algorithm 
from swarm_rescue.solutions.mapper_aristide import Mapper_Aristide
from swarm_rescue.solutions.explorer_aristide import Explorer_Aristide
from swarm_rescue.solutions.path_follower import PathFollower
from swarm_rescue.solutions.navigation import Navigator

class RescueDrone(DroneAbstract):
    def __init__(self, *, 
                 planner:GlobalPlanner, 
                 estimator:MeasuredState, 
                 semantic_state:SemanticState, 
                 controller:PDController,
        # PM TODO : Armand's algorithm 
                #  mapper:LidarMapper,
                #  explorer:Explorer,
        # PM TODO : Aristide's algorithm 
                 mapper:Mapper_Aristide,
                 explorer:Explorer_Aristide,
                 path_follower:PathFollower,
                 navigator: Navigator,
                 **kwargs):
        super().__init__(**kwargs)
        self.planner = planner
        self.estimator = estimator
        self.semantic_state = semantic_state
        self.controller = controller
        self.mapper = mapper
        self.explorer = explorer
        self.path_follower = path_follower
        self.navigator = navigator

        if self.mapper is not None and self.size_area is not None:
            self.mapper.set_map_size(self.size_area)

        self.debug_draw = DebugDraw(self.planner.mission, mapper)
        
    def define_message_for_all(self) -> None:
        return None

    def control(self) -> CommandsDict:

        # KALMAN filter
        self.estimator.update_estimate(self.get_GPS(), self.get_yaw(), self.get_velocity())

        # Look for wounded person on the way
        self.semantic_state.update_semantic(self.estimator, self.semantic_values(), self.grasper.grasped_wounded_persons)

        self.planner.update_mission(self.semantic_state.rescue_center_position, self.semantic_state, self.grasper.grasped_wounded_persons)
        
        # if isinstance(self.planner.mission, ExplorationMission):
        #TODO: replace with a getter, so that we can apply kalman filtering later
        ranges = self.lidar_values() 
        angles = self.lidar_rays_angles() if ranges is not None else None

        # # PM TODO : Armand's mapper algorithm 
        # self.mapper.update_lidar_map(self.estimator.position_, self.estimator.yaw_, ranges, angles)
        # self.mapper.update_occupancy_grid(self.estimator.position_, self.estimator.yaw_, ranges, angles, self.elapsed_timestep)
        
        # # PM TODO : remove before deliver - use of true position / compass
        self.mapper.update_lidar_map(self.true_position(), self.true_angle(), ranges, angles)
        self.mapper.update_occupancy_grid(self.true_position(), self.true_angle(), ranges, angles, self.elapsed_timestep)
        if self.elapsed_timestep % 10 == 0:
            self.debug_draw.plot_occupancy_grid(
                self.navigator.path_planner,
                start=self.estimator.position_,
                goal=self.semantic_state.rescue_center_position,
            )

        # PM TODO : Aristide's mapper algorithm 
        #self.explorer.scan(ranges, angles, self.estimator.position_, self.estimator.yaw_)

        decision = self.planner.decide(
            self.estimator, self.semantic_state, self.mapper, self.explorer,
            self.path_follower, navigator=self.navigator,
        )

        if decision.motion_reference is None or decision.motion_reference.target is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": decision.grasper}

        # Direct-to-target flight until a working Navigator is introduced.
        return self.controller.control(decision, self.estimator)

    def get_GPS(self):
        """Update GPS
        
        Returns:
            np.array(): x,y GPS position
        """
        return self.measured_gps_position()
        #return np.array(self.true_position(), dtype=float)

    def get_yaw(self):
        """Update yaw

        Returns:
            float: the yaw in radians
        """
        return self.measured_compass_angle()
        #return float(self.true_angle())

    def get_velocity(self):
        """Update velocity
        
        Returns:
            np.array(): x,y velocity
        """
        return self.measured_velocity()
        #return np.array(self.true_velocity(), dtype=float)

    def update_angular_velocity(self):
        """Update angular velocity

        Returns:
            float: the angular velocity in radians per second
        """
        return self.measured_angular_velocity()
        #return float(self.true_angular_velocity())

    # def draw_top_layer(self):
    #     self.debug_draw.draw_top_layer()

    # def plot_lidar_map(self):
    #     self.debug_draw.plot_lidar_map()
