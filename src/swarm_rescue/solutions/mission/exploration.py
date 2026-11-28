import numpy as np
from skimage.measure import label, regionprops

from ..data_types import WorldView


class FrontierExplorer:
    def __init__(self, minimum_cluster_size: int, minimum_distance: float):
        self.minimum_cluster_size = minimum_cluster_size
        self.minimum_distance = minimum_distance

    def find_frontier_clusters(self, mask, minimum_cluster_size=1):
        """Return connected frontier regions as arrays of grid coordinates.

        We have a minimum cluster size to avoid tagging one-off missed pixels.
        #TODO: harmonise our min cluster size with some constant?
        """
        if mask is None:
            return []

        # assign a cluster number
        cluster_labels = label(
            mask,
            background=0, # ignore non frontier cells
            connectivity=2, # allow edges and corner neighbours
        )

        # return the coords of cells making up each cluster
        return [
            region.coords
            for region in regionprops(cluster_labels)
            if region.area >= minimum_cluster_size
        ]

    def frontier_waypoint(self, grid, cluster, drone_position, states, minimum_distance=50.0,):
        """Choose an adjacent free waypoint, or None if none is eligible."""
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


    def candidates(
        self,
        world: WorldView,
        drone_position: np.ndarray,
    ):
        """Return one eligible world-coordinate waypoint per frontier cluster."""
        clusters = self.find_frontier_clusters(world.frontiers, self.minimum_cluster_size)

        if not clusters:
            return np.empty((0, 2), dtype=float)

        # reuse the same cell classification for every cluster
        states = world.grid.classify_cells()

        waypoints = []

        # calculate our frontier waypoints candidates
        for cluster in clusters:
            waypoint = self.frontier_waypoint(
                world.grid,
                cluster,
                drone_position,
                states,
                minimum_distance=self.minimum_distance,
            )
            if waypoint is not None:
                waypoints.append(waypoint)

        return np.asarray(waypoints, dtype=float).reshape(-1, 2)
