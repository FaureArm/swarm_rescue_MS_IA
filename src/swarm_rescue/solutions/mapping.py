import numpy as np


class LidarMapper:
    def __init__(self):
        self.map = set() # map holding hard pixels. Set to easily handle duplicates

    def update_lidar_map(self, drone_position, yaw, ranges, angles):
        """updates the drone's internal map by adding pixels and their immediate
        surrounding pixels as hard pixels in the drone's map. Hard limit of 275 px rays
        NB: could make a better version where rays calculate map-frame coords, drones share these
        which allows us to kalman the shit out of each hard pixel hit to have high confidence in hard px

        Args:
            drone_position (_type_): drone position in worl frame
            yaw (_type_): drone heading in world frame, in radian
        """
        # likely no situation where no lidar values are returned 
        # but just in case -> unless some filtering?
        # Sensor reads are supplied by RescueDrone at the original call site.
        if ranges is None:
            return

        ranges = np.asarray(ranges)
        angles = np.asarray(angles)

        # limit max ray distance to 275 px, because noise beyond that can make too many errors
        # to change later on if we get better readings
        valid = np.isfinite(ranges) & (ranges > 0) & (ranges <= 275)

        bearings = yaw + angles[valid]
        # give direction of px hit in a vector: [0, 1] for 90 deg hit for ex
        directions = np.column_stack((np.cos(bearings), np.sin(bearings)))
        # calculate the world frame coords of that hit (direction * length of ray + add drone's world frame pos)
        hits = drone_position + ranges[valid, None] * directions
        pixels = np.rint(hits).astype(int)

        for x, y in pixels:
            # add immediate neighbours to our positive pixel hit
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    self.map.add((x + dx, y + dy)) #adds coords of pixel. Set auto handles dupes

