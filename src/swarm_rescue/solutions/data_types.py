"""Shared data structures for the drone's components."""

from swarm_rescue.solutions.contracts.goals import MissionDecision, Waypoint
from swarm_rescue.solutions.mapping.occupancy_grid import OccupancyGrid

__all__ = ["MissionDecision", "Waypoint", "OccupancyGrid"]
