from skimage.measure import label, regionprops

from swarm_rescue.solutions.mapping.interface import MapSource


class FrontierDetector:
    #FRONTIER_UPDATE_INTERVAL = 100

    def __init__(self, map_source: MapSource, frontier_update_interval=100):
        self.map_source = map_source
        self.map_generation = map_source.map_generation
        self.frontier_mask_ = None # hold frontier masks for effenciency
        self.frontier_update_interval_ = frontier_update_interval
        self.frontier_revision = 0

    def update(self, timestep: int) -> None:
        if self.map_generation != self.map_source.map_generation:
            self.frontier_mask_ = None
            self.map_generation = self.map_source.map_generation

        # update frontier mask
        if self.frontier_mask_ is None or timestep % self.frontier_update_interval_ == 0:
            self.frontier_mask_ = self.map_source.occupancy_grid_.frontier_mask()
            self.frontier_revision += 1

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
