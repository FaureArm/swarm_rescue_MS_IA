from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from typing import TYPE_CHECKING

from swarm_rescue.solutions.contracts.observations import LidarObservation, StateObservation
from swarm_rescue.solutions.control.interface import MotionController
from swarm_rescue.solutions.decision.exploration.interface import ExplorationPolicy
from swarm_rescue.solutions.decision.interface import Behavior
from swarm_rescue.solutions.estimation.interface import StateEstimator
from swarm_rescue.solutions.mapping.interface import Mapper
from swarm_rescue.solutions.navigation.interface import Navigator
from swarm_rescue.solutions.visualization.debug_draw import DebugDraw

if TYPE_CHECKING:
    from swarm_rescue.solutions.assembly import ComponentFactory


class RescueDrone(DroneAbstract):
    def __init__(
        self, *, estimator: StateEstimator | None = None, mission: Behavior | None = None,
        controller: MotionController | None = None, mapper: Mapper | None = None,
        navigator: Navigator | None = None, exploration: ExplorationPolicy | None = None,
        component_factory: "ComponentFactory | None" = None, **kwargs,
    ):
        super().__init__(**kwargs)
        if component_factory is not None:
            if any(component is not None for component in (
                estimator, mission, controller, mapper, navigator, exploration
            )):
                raise ValueError("Supply a component factory or component instances, not both")
            components = component_factory(map_size=self.size_area)
            estimator = components["estimator"]
            mission = components["mission"]
            controller = components["controller"]
            mapper = components["mapper"]
            navigator = components["navigator"]
            exploration = components["exploration"]
        elif mapper is not None and self.size_area is not None:
            if mapper.map_size_ != self.size_area:
                mapper.set_map_size(self.size_area)

        if estimator is None or mission is None or controller is None or navigator is None:
            raise ValueError("Estimator, behavior, controller and navigator are required")
        self.estimator = estimator
        self.mission = mission
        self.controller = controller
        self.mapper = mapper
        self.navigator = navigator
        self.exploration = exploration
        self.debug_draw = DebugDraw(mission, mapper, exploration)
        # NB: later could add attributes to define role -> done in mission planner, 

    def define_message_for_all(self) -> None:
        return None

    def control(self) -> CommandsDict:
        # Mapping experiments continue using the existing true-pose inputs.
        if self.mapper is not None:
            drone_position = self.true_position() #self.get_GPS()
            yaw = self.true_angle() #self.get_yaw()
            ranges = self.lidar_values() #TODO: replace with a getter, so that we can apply kalman filtering later
            angles = self.lidar_rays_angles() if ranges is not None else None
            self.mapper.update(LidarObservation(
                drone_position, yaw, ranges, angles, self.elapsed_timestep
            ))
        if self.exploration is not None:
            self.exploration.update(self.elapsed_timestep)

        # keep updating pose and such, and applying safety if no measures
        # TODO: Keep this in a dedicated function to repeat it more often?
        estimate = self.estimator.update(StateObservation(
            self.get_GPS(), self.get_yaw(), self.get_velocity(), self.elapsed_timestep
        ))
        if (
            estimate is None
            or not estimate.is_available
        ):
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        decision = self.mission.decide(estimate.position)

        # update occupancy grid plot
        if (
            self.mapper is not None
            and self.identifier == 0
            and self.elapsed_timestep % 10 == 0
        ):
            self.debug_draw.plot_occupancy_grid()

        # Direct-to-target flight until a working Navigator is introduced.
        reference = self.navigator.navigate(decision.destination, estimate)
        if reference is None:
            return {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}

        command = self.controller.compute_commands(reference, estimate)
        return {
            "forward": command.forward,
            "lateral": command.lateral,
            "rotation": command.rotation,
            "grasper": decision.grasper,
        }

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
