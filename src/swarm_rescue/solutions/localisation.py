import numpy as np
from swarm_rescue.solutions.kalman import KalmanFusion


class MeasuredState:
    def __init__(self):
        self._kf = KalmanFusion()
        self.position_ = None
        self.yaw_ = None
        self.velocity_ = np.zeros(2)

    def update_estimate(self, position, yaw, velocity):
        """Store the GPS, compass and velocity readings for this tick."""
        self.yaw_ = yaw

        # Application filtre Kalman
        est = self._kf.step(position, velocity)
        self.position_ = np.asarray(est[:2])
        self.velocity_ = np.asarray(self._kf.velocity())
        return True
