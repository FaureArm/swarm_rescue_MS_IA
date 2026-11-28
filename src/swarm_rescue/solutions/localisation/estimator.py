import numpy as np
from .kalman import KalmanFusion
from ..data_types import StateEstimate
from .interface import EstimationContext



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
        self.position_ = None if est is None else np.asarray(est[:2])
        self.velocity_ = None if est is None else np.asarray(self._kf.velocity())
        return est is not None

    def update(self, drone: EstimationContext) -> StateEstimate | None:
        readings = drone.tick_observations
        self.update_estimate(readings.position, readings.yaw, readings.velocity)
        if self.position_ is None or self.yaw_ is None or not np.isfinite(self.yaw_):
            return None
        return StateEstimate(self.position_, self.yaw_, self.velocity_)
