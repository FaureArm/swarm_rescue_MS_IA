import numpy as np
from swarm_rescue.solutions import data_types as dt
from skimage.draw import line

class LidarMapper:
    def __init__(self):
        self.lidar_map_= set() # map holding hard pixels. Set to easily handle duplicates
        self.occupancy_grid_ = dt.OccupancyGrid() # occupancy grid to hold probabilistic map #TODO initalise with max map size
        self.max_lidar_range_ = 275 # max lidar range in pixelsS
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



    def update_lidar_map(self, drone_position, yaw, ranges, angles):
        """updates the drone's internal map by adding pixels and their immediate
        surrounding pixels as hard pixels in the drone's map. Hard limit of 275 px rays
        NB: could make a better version where rays calculate map-frame coords, drones share these
        which allows us to kalman the shit out of each hard pixel hit to have high confidence in hard px

        Args:
            drone_position (_type_): drone position in worl frame
            yaw (_type_): drone heading in world frame, in radian
        """
        pixels = self.get_lidar_hits(drone_position, yaw, ranges, angles)

        for x, y in pixels:
            # add immediate neighbours to our positive pixel hit
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.lidar_map_.add((x + dx, y + dy)) #adds coords of pixel. Set auto handles dupes

    def update_occupancy_grid(self, drone_position, yaw, ranges, angles, timestep):
        """_summary_

        """
        # we know positive hits
        pixels = self.get_lidar_hits(drone_position, yaw, ranges, angles)
        # however, we also want to use the fact that in-between pixels
        # were empty, so we need to calculate a line between our drone and the
        # pixels positive hits


        for pixel in pixels:
            # calculate pixels composing the line between drone
            # and pixel hit
            # TODO: uses a very basic line drawing algo, can be upgraded, but see if necessary
            drone_x, drone_y = np.rint(drone_position).astype(int)
            rows, cols = line(drone_y, drone_x, pixel[1], pixel[0])

            # Free pixels: exclude the endpoint
            self.occupancy_grid_.update_log_odds(cols[:-1], rows[:-1], pos_hit=False)

            # Occupied pixel: endpoint only
            self.occupancy_grid_.update_log_odds(cols[-1], rows[-1], pos_hit=True)
            # TODO: update timestep