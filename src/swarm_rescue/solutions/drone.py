from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.solutions.debug_draw import DebugDraw


class RescueDrone(DroneAbstract):
    def __init__(self, *, estimator, mission, controller, mapper=None, **kwargs):
        super().__init__(**kwargs)
        self.estimator = estimator
        self.mission = mission
        self.controller = controller
        self.mapper = mapper
        if self.mapper is not None and self.size_area is not None:
            self.mapper.set_map_size(self.size_area)
        self.debug_draw = DebugDraw(mission, mapper)
        # NB: later could add attributes to define role

    def define_message_for_all(self) -> None:
        return None

    def control(self) -> CommandsDict:
        # Preserve the mapping test's true pose inputs and read order, before
        # waypoint selection and measured-pose updates. The guard is unchanged.
        if self.mapper is not None:
            drone_position = self.true_position()
            yaw = self.true_angle()
            ranges = self.lidar_values() #TODO: replace with a getter, so that we can apply kalman filtering later
            angles = self.lidar_rays_angles() if ranges is not None else None
            self.mapper.update_lidar_map(drone_position, yaw, ranges, angles)
            self.mapper.update_occupancy_grid(drone_position, yaw, ranges, angles, self.elapsed_timestep)
            # only show occupancy grid map for a single drone
            if self.identifier == 0:
                self.debug_draw.plot_occupancy_grid()

        decision = self.mission.decide(self.estimator.position_)

        # Keep the original no-goal short circuit and direct sensor estimates.
        if decision.destination is None or not self.estimator.update_estimate(
            self.get_GPS(), self.get_yaw(), self.get_velocity()
        ):
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

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
        self.debug_draw.draw_top_layer()

    def plot_lidar_map(self):
        self.debug_draw.plot_lidar_map()
