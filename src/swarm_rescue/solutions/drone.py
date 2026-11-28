from __future__ import annotations

from typing import TYPE_CHECKING

from swarm_rescue.simulation.drone.controller import CommandsDict
from swarm_rescue.simulation.drone.drone_abstract import DroneAbstract
from swarm_rescue.simulation.ray_sensors.drone_semantic_sensor import DroneSemanticSensor

from .data_types import Detection, NavigationResult, ObjectKind, Observations

if TYPE_CHECKING:
    from .assembly import ComponentFactory

_DETECTION_KINDS = {
    DroneSemanticSensor.TypeEntity.WOUNDED_PERSON: ObjectKind.WOUNDED,
    DroneSemanticSensor.TypeEntity.RESCUE_CENTER: ObjectKind.RESCUE_CENTER,
}


class RescueDrone(DroneAbstract):
    def __init__(self, *, build_components: ComponentFactory, **kwargs):
        super().__init__(**kwargs)
        self._build_components = build_components
        self._initialize_components()

    def _initialize_components(self):
        self.parts = self._build_components(self.size_area)
        self.navigation_result = NavigationResult()

    def reset(self) -> None:
        super().reset()
        self._initialize_components()

    def define_message_for_all(self) -> None:
        return None

    def control(self) -> CommandsDict:
        parts = self.parts
        self.tick_observations = self.read_observations()
        self.pose = parts.localisation.update(self)
        self.world = parts.world_model.update(self)
        self.intent = parts.mission.update(self)
        self.navigation_result = parts.navigator.update(self)
        commands = parts.controller.update(self)
        if parts.visualization is not None:
            parts.visualization.update(self)
        return commands

    def read_observations(self) -> Observations:
        detections = self.semantic_values()
        observed_objects = tuple(
            Detection(_DETECTION_KINDS[item.entity_type], item.distance, item.angle)
            for item in (() if detections is None else detections)
            if item.entity_type in _DETECTION_KINDS
        )
        ranges = self.lidar_values()
        return Observations(
            self.elapsed_timestep, self.get_GPS(), self.get_yaw(), self.get_velocity(),
            ranges, self.lidar_rays_angles() if ranges is not None else None,
            observed_objects, bool(self.grasped_wounded_persons()),
        )

    def get_GPS(self):
        return self.measured_gps_position()

    def get_yaw(self):
        return self.measured_compass_angle()

    def get_velocity(self):
        return self.measured_velocity()
