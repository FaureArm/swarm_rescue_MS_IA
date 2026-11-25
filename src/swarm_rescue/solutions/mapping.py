import numpy as np
from swarm_rescue.solutions import data_types as dt
from skimage.draw import line
from skimage.measure import label, regionprops

class LidarMapper:

    #FRONTIER_UPDATE_INTERVAL = 100

    def __init__(self, map_size = (2000, 2000), frontier_update_interval = 100):
        self.lidar_map_= set() # map holding hard pixels. Set to easily handle duplicates
        self.occupancy_grid_ = dt.OccupancyGrid(*map_size)
        self.max_lidar_range_ = 275 # max lidar range in pixelsS
        self.map_size_ = map_size
        self.frontier_mask_ = None
        self.frontier_waypoints_ = np.empty((0, 2), dtype=float)
        self.frontier_update_interval_ = frontier_update_interval
        self.rescue_center_position = None

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

    def get_lidar_hits(self, drone_position, yaw, ranges, angles):
        """get the lidar hits, as pixels. Only returns positive hits !
        """
        if ranges is None:
            return np.empty((0, 2), dtype=int)

        ranges = np.asarray(ranges)
        angles = np.asarray(angles)
        # for now we set a max range, as recommended by the readme because
        # of noisy values. Later on, with kalman filtering, we can increase this range and filter out the noise
        valid = np.isfinite(ranges) & (ranges > 0) & (ranges <= self.max_lidar_range_)

        bearings = yaw + angles[valid]
        directions = np.column_stack((np.cos(bearings), np.sin(bearings)))
        hits = drone_position + ranges[valid, None] * directions
        pixels = np.rint(hits).astype(int)

        return pixels

    def get_lidar_empties(self, drone_position, yaw, ranges, angles):
        if ranges is None:
            return np.empty((0, 2), dtype=int)

        ranges = np.asarray(ranges)
        angles = np.asarray(angles)
        # we get rays that are of beyond the max range
        valid = np.isfinite(ranges) & (ranges > self.max_lidar_range_)

        bearings = yaw + angles[valid]
        directions = np.column_stack((np.cos(bearings), np.sin(bearings)))
        empties = drone_position + self.max_lidar_range_ * directions
        return np.rint(empties).astype(int)


    def update_lidar_map(self, drone_position, yaw, ranges, angles):
        """updates the drone's internal map by adding pixels and their immediate
        surrounding pixels as hard pixels in the drone's map. Hard limit of 275 px rays
        NB: could make a better version where rays calculate map-frame coords, drones share these
        which allows us to kalman the shit out of each hard pixel hit to have high confidence in hard px

        Args:
            drone_position (_type_): drone position in worl frame
            yaw (_type_): drone heading in world frame, in radian
        """
        # PM : store rescue center position at first shot
        if (self.rescue_center_position is None and 
            drone_position is not None):
            self.rescue_center_position = np.asarray(drone_position)

        pixels = self.get_lidar_hits(drone_position, yaw, ranges, angles)

        for x, y in pixels:
            # add immediate neighbours to our positive pixel hit
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.lidar_map_.add((x + dx, y + dy)) #adds coords of pixel. Set auto handles dupes

    def update_occupancy_grid(self, drone_position, yaw, ranges, angles, timestep):
        """Update ray evidence and record the simulation timestep for observed cells."""

        drone_x, drone_y = np.rint(drone_position).astype(int)

        # we know positive hits
        pixels = self.get_lidar_hits(drone_position, yaw, ranges, angles)
        # however, we also want to use the fact that in-between pixels
        # were empty, so we need to calculate a line between our drone and the
        # pixels positive hits

        for pixel in pixels:
            # calculate pixels composing the line between drone
            # and pixel hit
            # TODO: uses a very basic line drawing algo, can be upgraded, but see if necessary
            
            rows, cols = line(drone_y, drone_x, pixel[1], pixel[0])

            # Free pixels: exclude the endpoint
            self.occupancy_grid_.update_log_odds(
                cols[:-1], rows[:-1], pos_hit=False, timestep=timestep,
            )

            # Occupied pixel: endpoint only
            self.occupancy_grid_.update_log_odds(
                cols[-1], rows[-1], pos_hit=True, timestep=timestep,
            )
        

        # we also need to check empty spaces
        empty_endpoints = self.get_lidar_empties(drone_position, yaw, ranges, angles)
        for end_x, end_y in empty_endpoints:
            rows, cols = line(drone_y, drone_x, end_y, end_x)

            self.occupancy_grid_.update_log_odds(
                cols, rows, pos_hit=False, timestep=timestep,
            )

        # update frontier mask
        if self.frontier_mask_ is None or timestep % self.frontier_update_interval_ == 0:
            self.frontier_mask_ = self.occupancy_grid_.frontier_mask()


    def find_frontier_clusters(self, minimum_cluster_size=1):
        """Return connected frontier regions as arrays of grid coordinates.

        We have a minimum cluster size to avoid tagging one-off missed pixels.
        #TODO: harmonise our min cluster size with some constant?
        """
        if self.frontier_mask_ is None:
            return []

        # assign a cluster number
        cluster_labels = label(
            self.frontier_mask_,
            background=0, # ignore non frontier cells
            connectivity=2, # allow edges and corner neighbours
        )

        # return the coords of cells making up each cluster
        return [
            region.coords
            for region in regionprops(cluster_labels)
            if region.area >= minimum_cluster_size
        ]

    def frontier_waypoint(self, cluster, drone_position, states, minimum_distance=50.0,):
        """Choose an adjacent free waypoint, or None if none is eligible."""
        grid = self.occupancy_grid_

        # check if we have new frontier cells detected since detection
        cluster = cluster[states[cluster[:, 0], cluster[:, 1]] == grid.UNKNOWN]
        if len(cluster) == 0:
            return None

        free_neighbors = []

        # check neighbours of each frontier cell to check if its free
        for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbors = cluster + offset

            in_bounds = (
                (neighbors[:, 0] >= 0)
                & (neighbors[:, 0] < grid.width)
                & (neighbors[:, 1] >= 0)
                & (neighbors[:, 1] < grid.height)
            )
            neighbors = neighbors[in_bounds]

            is_free = states[neighbors[:, 0], neighbors[:, 1]] == grid.FREE
            free_neighbors.append(neighbors[is_free])

        # get rid of duplicates, some frontier cells might be 
        # sharing the same free cells candidatess
        candidates = np.unique(np.concatenate(free_neighbors), axis=0)

        # set back to occupancy grid origin and switch to
        # world coords
        origin = np.array([grid.width // 2, grid.height // 2])
        world_candidates = (candidates - origin).astype(float)

        # get rid of exploration candidates too close to the drone
        # we can safely assume they are not worth checking
        distances = np.linalg.norm(world_candidates - drone_position, axis=1)
        world_candidates = world_candidates[distances >= minimum_distance]

        if len(world_candidates) == 0:
            return None

        # actual free cell candidate selection, using the centroid of
        # a cluster as reference, and then taking the closest one
        cluster_center = cluster.mean(axis=0) - origin
        squared_distances = np.sum((world_candidates - cluster_center) ** 2, axis=1)

        return world_candidates[np.argmin(squared_distances)]


    def update_frontier_waypoints(
        self,
        drone_position,
        minimum_cluster_size=5,
        minimum_distance=10.0,
    ):
        """Cache one eligible world-coordinate waypoint per frontier cluster."""
        clusters = self.find_frontier_clusters(minimum_cluster_size=minimum_cluster_size)
        
        if not clusters:
            self.frontier_waypoints_ = np.empty((0, 2), dtype=float)
            return self.frontier_waypoints_

        # reuse the same cell classification for every cluster
        states = self.occupancy_grid_.classify_cells()

        waypoints = []

        # calculate our frontier waypoints candidates
        for cluster in clusters:
            waypoint = self.frontier_waypoint(
                cluster,
                drone_position,
                states,
                minimum_distance=minimum_distance,
            )
            if waypoint is not None:
                waypoints.append(waypoint)

        self.frontier_waypoints_ = np.asarray(waypoints, dtype=float).reshape(-1, 2)
        return self.frontier_waypoints_

    def select_exploration_waypoint(
        self,
        drone_position,
        minimum_cluster_size=5,
        minimum_distance=50.0,
    ):
        """Refresh candidates and return a copy of the nearest waypoint, or None."""
        waypoints = self.update_frontier_waypoints(
            drone_position,
            minimum_cluster_size=minimum_cluster_size,
            minimum_distance=minimum_distance,
        )

        if len(waypoints) == 0:
            return None

        # so far we use distance to determine the waypoint we choose
        #TODO: try other methods
        distances = np.linalg.norm(waypoints - drone_position, axis=1)
        return waypoints[np.argmax(distances)].copy()
