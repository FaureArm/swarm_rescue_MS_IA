import numpy as np


class MeasuredState:
    def __init__(self):
        self.position_ = None
        self.yaw_ = None
        self.velocity_ = np.zeros(2)

    def update_estimate(self, position, yaw, velocity):
        """Store the GPS, compass and velocity readings for this tick."""
        self.position_ = position
        self.yaw_ = yaw
        self.velocity_ = velocity
        return True
