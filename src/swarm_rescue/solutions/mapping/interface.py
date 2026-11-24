from typing import Protocol

from swarm_rescue.solutions.contracts.observations import LidarObservation
from swarm_rescue.solutions.mapping.occupancy_grid import OccupancyGrid


class MapSource(Protocol):
    """Queries share one grid per drone; consumers do not modify its arrays.

    map_generation changes when the grid is replaced. Consumers refresh their
    caches on their next update, before decisions or visualization run.
    """

    @property
    def occupancy_grid_(self) -> OccupancyGrid: ...

    @property
    def map_generation(self) -> int: ...


class Mapper(MapSource, Protocol):
    @property
    def lidar_map_(self) -> set[tuple[int, int]]: ...

    @property
    def map_size_(self) -> tuple[int, int]: ...

    def update(self, observation: LidarObservation) -> None: ...

    def set_map_size(self, map_size: tuple[int, int]) -> None: ...
