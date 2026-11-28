from ..data_types import WorldView
from .interface import Mapping, ObjectTracking, WorldContext


class WorldModel:
    def __init__(self, mapper: Mapping, tracker: ObjectTracking):
        self._mapper = mapper
        self._tracker = tracker
        self._rescue_center = None

    def update(self, drone: WorldContext) -> WorldView:
        readings = drone.tick_observations
        pose = drone.pose
        wounded = ()
        if pose is not None:
            self._mapper.integrate(pose, readings.ranges, readings.bearings, readings.timestep)
            wounded, self._rescue_center = self._tracker.update(pose, readings.detections)
        return WorldView(
            self._mapper.grid, self._mapper.frontiers,
            self._mapper.frontier_revision, wounded, self._rescue_center,
        )
