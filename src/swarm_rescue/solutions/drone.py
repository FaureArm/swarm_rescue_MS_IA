from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.solutions.debug_draw import DebugDraw
from swarm_rescue.solutions.localisation import MeasuredState, SemanticState
from swarm_rescue.solutions.mission_planner import (GlobalPlanner, 
                                                    ExplorationMission)
from swarm_rescue.solutions.control import PDController
from swarm_rescue.solutions.mapping import LidarMapper


class RescueDrone(DroneAbstract):
    def __init__(self, *, 
                 planner:GlobalPlanner, 
                 estimator:MeasuredState, 
                 semantic_state:SemanticState, 
                 controller:PDController, 
                 mapper:LidarMapper, 
                 **kwargs):
        super().__init__(**kwargs)
        self.planner = planner
        self.estimator = estimator
        self.semantic_state = semantic_state
        self.controller = controller
        self.mapper = mapper

        if self.mapper is not None and self.size_area is not None:
            self.mapper.set_map_size(self.size_area)

        self.debug_draw = DebugDraw(self.planner.mission, mapper)
        # NB: later could add attributes to define role -> done in mission planner, 

    def define_message_for_all(self) -> None:
        return None

    def control(self) -> CommandsDict:
        # keep updating pose and such, and applying safety if no measures
        self.estimator.update_estimate(
            self.get_GPS(), self.get_yaw(), self.get_velocity())

        self.semantic_state.process_semantic_data(self.estimator, self.semantic_values())

        self.planner.update_mission(self.estimator.position_, self.semantic_state)
        
        if isinstance(self.planner.mission, ExplorationMission):
            #TODO: replace with a getter, so that we can apply kalman filtering later
            ranges = self.lidar_values() 
            angles = self.lidar_rays_angles() if ranges is not None else None
            self.mapper.update_lidar_map(self.estimator.position_, self.estimator.yaw_, ranges, angles)
            self.mapper.update_occupancy_grid(self.estimator.position_, self.estimator.yaw_, ranges, angles, self.elapsed_timestep)

        decision = self.planner.decide(self.estimator, self.semantic_state, self.mapper)

        if decision.destination is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        # update occupancy grid plot
        if (
            self.mapper is not None
        ):
            self.debug_draw.plot_occupancy_grid()

        # Direct-to-target flight until a working Navigator is introduced.
        return self.controller.control(
            decision.destination, self.estimator, decision.grasper
        )

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

    def draw_top_layer(self):
        return
        # self.debug_draw.draw_top_layer()

    def plot_lidar_map(self):
        return
        # self.debug_draw.plot_lidar_map()
