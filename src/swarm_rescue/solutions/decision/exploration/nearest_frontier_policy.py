import numpy as np

from swarm_rescue.solutions.decision.exploration.frontier_detector import FrontierDetector


class NearestFrontierPolicy:
    def __init__(self, detector: FrontierDetector):
        self.detector = detector
        self.frontier_waypoints_ = np.empty((0, 2), dtype=float)

    @property
    def occupancy_grid_(self):
        return self.detector.map_source.occupancy_grid_

    @property
    def frontier_mask_(self):
        return self.detector.frontier_mask_

    @property
    def frontier_revision(self) -> int:
        return self.detector.frontier_revision

    def update(self, timestep: int) -> None:
        previous_generation = self.detector.map_generation
        self.detector.update(timestep)
        if previous_generation != self.detector.map_generation:
            self.frontier_waypoints_ = np.empty((0, 2), dtype=float)

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
        clusters = self.detector.find_frontier_clusters(minimum_cluster_size=minimum_cluster_size)
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
        return waypoints[np.argmin(distances)].copy()
