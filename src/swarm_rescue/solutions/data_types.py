"""Shared data structures for the drone's components."""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

from .world.grid import OccupancyGrid as OccupancyGrid

class Activity(Enum):
    """
    All the states of the drone as a state machine
    """
    STAND_BY = -1
    EXPLORING = 0
    SEARCHING_WOUNDED = 1
    GRASPING_WOUNDED = 2
    SEARCHING_RESCUE_CENTER = 3
    DROPPING_AT_RESCUE_CENTER = 4


class MotionReference(NamedTuple):
    target: np.ndarray       # lookahead point, useful for display
    velocity: np.ndarray | None  # desired world-frame velocity, px/tick
    heading: float | None


class MissionDecision(NamedTuple):
    # small class to hold decisions and reuse easily
    motion_reference: MotionReference | None
    grasper: int
    role: Activity


class Waypoint(NamedTuple):
    """A position in the map's world coordinates, in pixels
    Attributes:
        x (float): The x-coordinate of the waypoint.
        y (float): The y-coordinate of the waypoint.
        velocity (float): The target velocity to have when reaching the waypoint
        heading (float): The target heading to have when reaching the waypoint
    """

    x: float
    y: float
    velocity: float
    heading: float | None # for now might not use heading due to holonomic nature


class ObjectKind(Enum):
    WOUNDED = "wounded"
    RESCUE_CENTER = "rescue_center"


class Detection(NamedTuple):
    kind: ObjectKind
    distance: float
    angle: float


@dataclass(frozen=True)
class Observations:
    """One tick of measurements. Positions and ranges in px, yaw in radians, velocity in px/tick."""

    timestep: int
    position: np.ndarray | None
    yaw: float | None
    velocity: np.ndarray | None
    ranges: np.ndarray | None
    bearings: np.ndarray | None
    detections: tuple[Detection, ...] = ()
    holding_wounded: bool = False


class StateEstimate(NamedTuple):
    """Complete pose in world coordinates and velocity in px/tick."""

    position: np.ndarray
    yaw: float
    velocity: np.ndarray


@dataclass(frozen=True)
class WorldView:
    """Borrowed map data. Consumers treat the grid and its arrays as read-only."""

    grid: OccupancyGrid
    frontiers: np.ndarray | None = None
    frontier_revision: int = 0
    wounded: tuple[np.ndarray, ...] = ()
    rescue_center: np.ndarray | None = None


class MovementMode(Enum):
    IDLE = "idle"
    DIRECT = "direct"
    PLANNED = "planned"


@dataclass(frozen=True)
class MissionIntent:
    """Mission output for one tick. Destinations are [x, y] world pixels."""

    activity: Activity
    destination: np.ndarray | None = None
    mode: MovementMode = MovementMode.IDLE
    grasper: int = 0
    face_target: bool = False
    candidates: np.ndarray | None = None


class NavigationStatus(Enum):
    IDLE = "idle"
    FOLLOWING = "following"
    ARRIVED = "arrived"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class NavigationResult:
    status: NavigationStatus = NavigationStatus.IDLE
    motion: MotionReference | None = None
    route: np.ndarray | None = None
    traversable: np.ndarray | None = None
