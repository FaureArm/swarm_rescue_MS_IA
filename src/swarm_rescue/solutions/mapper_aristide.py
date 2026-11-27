from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from skimage.draw import line

from swarm_rescue.solutions import data_types as dt

Cell = Tuple[int, int]
class Mapper_Aristide:

    def __init__(self, map_size = (2000, 2000), frontier_update_interval = 100):

        self.map: Dict[Cell, int] = {}
        self.room_map: Dict[Cell, int] = {}
        self.wall_clear_scans: Dict[Cell, int] = {}
        self.wall_confirm_scans: Dict[Cell, int] = {}
        self.contact_walls: Set[Cell] = set()
        self.room_contact_walls: Set[Cell] = set()


        self.lidar_map_= set() # map holding hard pixels. Set to easily handle duplicates
        self.occupancy_grid_ = dt.OccupancyGrid(*map_size)
        self.max_lidar_range_ = 275 # max lidar range in pixelsS
        self.map_size_ = map_size
        self.frontier_mask_ = None
        self.frontier_waypoints_ = np.empty((0, 2), dtype=float)
        self.frontier_update_interval_ = frontier_update_interval
        self.drone_spawn_position = None
        self.lidar_free_bearings = None


    #     # PM TODO : USELESS - From Armand's algorithm, to keep mutual code unchanged
    # just for POC
    def set_map_size(self, map_size):
        """Set the playground dimensions and recreate the grid before mapping starts."""
        self.map_size_ = map_size
        self.occupancy_grid_ = dt.OccupancyGrid(
            *map_size,
            pos_hit_increment=self.occupancy_grid_.log_odd_plus,
            neg_hit_increment=self.occupancy_grid_.log_odd_minus,
        )
        self.frontier_mask_ = None
        self.frontier_waypoints_ = np.empty((0, 2), dtype=float)
        